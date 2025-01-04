from torch.utils.data import DataLoader, Subset  
import torch.nn as nn  
import torch  
import pandas as pd  
import numpy as np  
from sklearn.model_selection import StratifiedKFold  
import torch.optim as optim   
from dataset import FMRIDataGenerator  
from CNNLSTM import CNNLSTM  
import config  
from tqdm import tqdm  
import os  
import matplotlib.pyplot as plt  
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc  
import seaborn as sns  
from itertools import cycle    

def create_folds():  
    dataset_dir = config.DATASET_DIR  
    file = pd.read_csv(config.METADATA_CSV)  
    list_IDs = file['Image']  
    labels = dict(zip(file['Image'], file['DX']))   
    dataset = FMRIDataGenerator(list_IDs=list_IDs, labels=labels, dataset_dir=dataset_dir)  
    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=42)  
    label_list = [labels[img_id] for img_id in list_IDs]  
    folds = []  
    fold_data = []  
    for fold, (train_idx, val_idx) in enumerate(skf.split(list_IDs, label_list)):  
        train_subset = Subset(dataset, train_idx)  
        val_subset = Subset(dataset, val_idx)  
        train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True)  
        val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False)  
        fold_data.append({  
            'fold': fold + 1,  
            'train_images': [list_IDs[i] for i in train_idx],  
            'train_labels': [labels[list_IDs[i]] for i in train_idx],  
            'val_images': [list_IDs[i] for i in val_idx],  
            'val_labels': [labels[list_IDs[i]] for i in val_idx]  
        })  
        folds.append({'train': train_loader, 'val': val_loader})  
    
    fold_df = pd.DataFrame(fold_data)  
    fold_df.to_csv('fold_data.csv', index=False)  
    return folds  

def save_checkpoint(state, fold_idx, epoch):  
    """保存检查点"""  
    checkpoint_dir = 'checkpoints'  
    os.makedirs(checkpoint_dir, exist_ok=True)  
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_fold_{fold_idx + 1}_epoch_{epoch}.pth')  
    torch.save(state, checkpoint_path)  
    print(f"Checkpoint saved: {checkpoint_path}")  

def load_checkpoint(checkpoint_path, model, optimizer):  
    """加载检查点"""  
    if os.path.exists(checkpoint_path):  
        checkpoint = torch.load(checkpoint_path)  
        model.load_state_dict(checkpoint['model_state_dict'])  
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
        start_epoch = checkpoint['epoch']  
        history = checkpoint['history']  
        print(f"Resuming from epoch {start_epoch}")  
        return start_epoch, history  
    return 0, {'train_loss': [], 'val_loss': [], 'val_acc': []}  


def plot_confusion_matrix(y_true, y_pred, fold_idx, prefix=''):  
    """绘制混淆矩阵"""  
    plt.figure(figsize=(8, 6))  
    cm = confusion_matrix(y_true, y_pred)  
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  
    plt.title(f'{prefix}Confusion Matrix (Fold {fold_idx + 1})')  
    plt.ylabel('True Label')  
    plt.xlabel('Predicted Label')
    save_path = os.path.join("/root/autodl-tmp/CNNLSTM/Project/fMRI/fig_output",f'{prefix}confusion_matrix_fold_{fold_idx + 1}.png')
    plt.savefig(save_path)  
    plt.close()  

def plot_roc_curve(y_true, y_scores, fold_idx, num_classes, prefix=''):  
    """绘制ROC曲线"""  
    plt.figure(figsize=(8, 6))  
    
    # 为每个类别计算ROC曲线和AUC  
    fpr = dict()  
    tpr = dict()  
    roc_auc = dict()  
    
    # 确保y_true是整数类型  
    y_true = np.array(y_true, dtype=np.int64)  
    
    # 转换为one-hot编码  
    y_true_onehot = np.zeros((len(y_true), num_classes))  
    for i in range(len(y_true)):  
        y_true_onehot[i, y_true[i]] = 1  
    
    for i in range(num_classes):  
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_scores[:, i])  
        roc_auc[i] = auc(fpr[i], tpr[i])  
    
    # 绘制所有ROC曲线  
    colors = cycle(['blue', 'red', 'green', 'yellow', 'purple'])  
    for i, color in zip(range(num_classes), colors):  
        plt.plot(fpr[i], tpr[i], color=color, lw=2,  
                label=f'ROC curve (class {i}) (AUC = {roc_auc[i]:0.2f})')  
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)  
    plt.xlim([0.0, 1.0])  
    plt.ylim([0.0, 1.05])  
    plt.xlabel('False Positive Rate')  
    plt.ylabel('True Positive Rate')  
    plt.title(f'{prefix}ROC Curves (Fold {fold_idx + 1})')  
    plt.legend(loc="lower right") 
    save_path = os.path.join("/root/autodl-tmp/CNNLSTM/Project/fMRI/fig_output",f'{prefix}roc_curve_fold_{fold_idx + 1}.png')
    plt.savefig(save_path)  
    plt.close()

def plot_training_results(history, fold_idx):  
    """绘制训练结果图"""  
    plt.figure(figsize=(15, 5))  
    
    # 绘制损失曲线  
    plt.subplot(1, 3, 1)  
    plt.plot(history['train_loss'], label='Training Loss')  
    plt.plot(history['val_loss'], label='Validation Loss')  
    plt.title(f'Loss Curves (Fold {fold_idx + 1})')  
    plt.xlabel('Epoch')  
    plt.ylabel('Loss')  
    plt.legend()  
    
    # 绘制准确率曲线  
    plt.subplot(1, 3, 2)  
    plt.plot(history['val_acc'], label='Validation Accuracy')  
    plt.title(f'Accuracy Curve (Fold {fold_idx + 1})')  
    plt.xlabel('Epoch')  
    plt.ylabel('Accuracy (%)')  
    plt.legend()  
    
    # 绘制F1分数曲线  
    plt.subplot(1, 3, 3)  
    plt.plot(history['val_f1'], label='Validation F1')  
    plt.title(f'F1 Score Curve (Fold {fold_idx + 1})')  
    plt.xlabel('Epoch')  
    plt.ylabel('F1 Score')  
    plt.legend()  
    
    plt.tight_layout()  
    save_path = os.path.join("/root/autodl-tmp/CNNLSTM/Project/fMRI/fig_output",f'training_results_fold_{fold_idx + 1}.png')
    plt.savefig(save_path)  
    plt.close()  

def evaluate_model(model, val_loader, criterion):  
    """评估模型性能"""  
    model.eval()  
    val_loss = 0.0  
    all_predictions = []  
    all_labels = []  
    all_scores = []  
    
    with torch.no_grad():  
        for X, y in val_loader:  
            X, y = X.cuda(), y.cuda()  
            outputs = model(X)  
            loss = criterion(outputs, y.long())  
            val_loss += loss.item()  
             
            probabilities = torch.softmax(outputs, dim=1)  
            _, predicted = torch.max(outputs, 1)  
            
            all_predictions.extend(predicted.cpu().numpy())  
            all_labels.extend(y.cpu().numpy())  
            all_scores.extend(probabilities.cpu().numpy())  
    
    all_predictions = np.array(all_predictions)  
    all_labels = np.array(all_labels)  
    all_scores = np.array(all_scores)  
    
    accuracy = (all_predictions == all_labels).mean() * 100  
    f1 = f1_score(all_labels, all_predictions, average='weighted')  
    avg_loss = val_loss / len(val_loader)  
    
    return {  
        'loss': avg_loss,  
        'accuracy': accuracy,  
        'f1': f1,  
        'predictions': all_predictions,  
        'true_labels': all_labels,  
        'scores': all_scores  
    }  

def cross_validate(folds, model, criterion):  
    """进行交叉验证评估"""  
    all_predictions = []  
    all_labels = []  
    all_scores = []  
    total_loss = 0.0  
    
    model.eval()  
    with torch.no_grad():  
        for fold in folds:  
            val_loader = fold['val']  
            for X, y in val_loader:  
                X, y = X.cuda(), y.cuda()  
                outputs = model(X)  
                loss = criterion(outputs, y.long())  
                total_loss += loss.item()  
                
                probabilities = torch.softmax(outputs, dim=1)  
                _, predicted = torch.max(outputs, 1)  
                
                all_predictions.extend(predicted.cpu().numpy())  
                all_labels.extend(y.cpu().numpy())  
                all_scores.extend(probabilities.cpu().numpy())  
    
    # 计算整体指标  
    all_predictions = np.array(all_predictions)  
    all_labels = np.array(all_labels)  
    all_scores = np.array(all_scores)  
    
    accuracy = (all_predictions == all_labels).mean() * 100  
    f1 = f1_score(all_labels, all_predictions, average='weighted')  
    avg_loss = total_loss / (len(folds) * len(val_loader))  
    
    return {  
        'cv_loss': avg_loss,  
        'cv_accuracy': accuracy,  
        'cv_f1': f1,  
        'predictions': all_predictions,  
        'true_labels': all_labels,  
        'scores': all_scores  
    }  

def plot_cv_results(cv_history, current_fold):  
    """绘制交叉验证结果"""  
    plt.figure(figsize=(15, 5))  
    
    # 绘制损失曲线  
    plt.subplot(1, 3, 1)  
    plt.plot(cv_history['cv_loss'], label='CV Loss')  
    plt.title('Cross-Validation Loss')  
    plt.xlabel('Fold')  
    plt.ylabel('Loss')  
    plt.legend()  
    
    # 绘制准确率曲线  
    plt.subplot(1, 3, 2)  
    plt.plot(cv_history['cv_accuracy'], label='CV Accuracy')  
    plt.title('Cross-Validation Accuracy')  
    plt.xlabel('Fold')  
    plt.ylabel('Accuracy (%)')  
    plt.legend()  
    
    # 绘制F1分数曲线  
    plt.subplot(1, 3, 3)  
    plt.plot(cv_history['cv_f1'], label='CV F1')  
    plt.title('Cross-Validation F1 Score')  
    plt.xlabel('Fold')  
    plt.ylabel('F1 Score')  
    plt.legend()  
    
    plt.tight_layout()  
    save_path = os.path.join("/root/autodl-tmp/CNNLSTM/Project/fMRI/fig_output",f'cv_results_after_fold_{current_fold + 1}.png')
    plt.savefig(save_path)   
    plt.close()  

def train_and_eval(folds, resume_from=None):  
    model = CNNLSTM(config.NUM_CLASSES)  
    model = model.cuda()  
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=config.LR)  
    
    # 用于记录交叉验证的历史  
    cv_history = {  
        'cv_loss': [],  
        'cv_accuracy': [],  
        'cv_f1': []  
    }  

    for fold_idx, fold in enumerate(folds):  
        print(f"\nFold {fold_idx + 1}/{len(folds)}")  
        train_loader = fold['train']  
        val_loader = fold['val']  
        
        if resume_from:  
            checkpoint_path = f'checkpoints/checkpoint_fold_{fold_idx + 1}_epoch_{resume_from}.pth'  
            start_epoch, history = load_checkpoint(checkpoint_path, model, optimizer)  
        else:  
            start_epoch = 0  
            history = {  
                'train_loss': [],   
                'val_loss': [],   
                'val_acc': [],  
                'val_f1': []  
            }  

        try:  
            for epoch in range(start_epoch, config.EPOCH):  
                print(f"Epoch {epoch + 1}/{config.EPOCH}")  
                
                # 训练阶段  
                model.train()  
                total_loss = 0.0  
                for X, y in tqdm(train_loader, desc="Training", unit="batch"):  
                    X, y = X.cuda(), y.cuda()  
                    optimizer.zero_grad()  
                    outputs = model(X)  
                    loss = criterion(outputs, y.long())  
                    loss.backward()  
                    optimizer.step()  
                    total_loss += loss.item()  

                avg_train_loss = total_loss / len(train_loader)  
                history['train_loss'].append(avg_train_loss)  
                print(f"Training Loss: {avg_train_loss:.4f}")  
                
                # 验证阶段  
                eval_results = evaluate_model(model, val_loader, criterion)  
                
                # 更新历史记录  
                history['val_loss'].append(eval_results['loss'])  
                history['val_acc'].append(eval_results['accuracy'])  
                history['val_f1'].append(eval_results['f1'])  
                
                print(f'Validation Loss: {eval_results["loss"]:.4f}, '  
                      f'Accuracy: {eval_results["accuracy"]:.2f}%, '  
                      f'F1 Score: {eval_results["f1"]:.4f}')  

                # 绘制评估图表  
                plot_training_results(history, fold_idx)  
                plot_confusion_matrix(  
                    eval_results['true_labels'],  
                    eval_results['predictions'],  
                    fold_idx  
                )  
                plot_roc_curve(  
                    eval_results['true_labels'],  
                    eval_results['scores'],  
                    fold_idx,  
                    config.NUM_CLASSES  
                )  

                # 保存检查点  
                checkpoint = {  
                    'epoch': epoch + 1,  
                    'model_state_dict': model.state_dict(),  
                    'optimizer_state_dict': optimizer.state_dict(),  
                    'history': history  
                }  
                save_checkpoint(checkpoint, fold_idx, epoch + 1)  

            # 当前fold训练完成后，进行交叉验证评估  
            print("\nPerforming cross-validation evaluation...")  
            cv_results = cross_validate(folds, model, criterion)  
            
            # 更新交叉验证历史  
            cv_history['cv_loss'].append(cv_results['cv_loss'])  
            cv_history['cv_accuracy'].append(cv_results['cv_accuracy'])  
            cv_history['cv_f1'].append(cv_results['cv_f1'])  
            
            # 打印交叉验证结果  
            print(f"\nCross-Validation Results after Fold {fold_idx + 1}:")  
            print(f"CV Loss: {cv_results['cv_loss']:.4f}")  
            print(f"CV Accuracy: {cv_results['cv_accuracy']:.2f}%")  
            print(f"CV F1 Score: {cv_results['cv_f1']:.4f}")  
            
            # 绘制交叉验证结果  
            plot_cv_results(cv_history, fold_idx)  
            
            # 绘制整体的混淆矩阵和ROC曲线  
            plot_confusion_matrix(  
                cv_results['true_labels'],  
                cv_results['predictions'],  
                fold_idx,  
                prefix='cv_'  
            )  
            plot_roc_curve(  
                cv_results['true_labels'],  
                cv_results['scores'],  
                fold_idx,  
                config.NUM_CLASSES,  
                prefix='cv_'  
            )  

        except KeyboardInterrupt:  
            print("\nTraining interrupted. Saving checkpoint...")  
            checkpoint = {  
                'epoch': epoch + 1,  
                'model_state_dict': model.state_dict(),  
                'optimizer_state_dict': optimizer.state_dict(),  
                'history': history,  
                'cv_history': cv_history  
            }  
            save_checkpoint(checkpoint, fold_idx, epoch + 1)  
            print("You can resume training later using the saved checkpoint.")  
            raise  

        # 保存最终模型  
        torch.save(model.state_dict(), f"model_fold_{fold_idx + 1}.pth")  
        print(f"Final model for Fold {fold_idx + 1} saved.")  
    
    # 训练完成后，保存最终的交叉验证结果  
    cv_results_df = pd.DataFrame({  
        'Fold': range(1, len(folds) + 1),  
        'CV_Loss': cv_history['cv_loss'],  
        'CV_Accuracy': cv_history['cv_accuracy'],  
        'CV_F1': cv_history['cv_f1']  
    })  
    cv_results_df.to_csv('cross_validation_results.csv', index=False)  
    print("\nFinal Cross-Validation Results saved to 'cross_validation_results.csv'")
     


if __name__ == "__main__":  
    os.makedirs('checkpoints', exist_ok=True)  
    folds = create_folds()  
    train_and_eval(folds, resume_from=None)  # resume_from=epoch