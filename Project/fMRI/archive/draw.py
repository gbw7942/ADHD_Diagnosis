# Description: This file is used to draw the confusion matrix, ROC curve, and training results.
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from itertools import cycle


def plot_confusion_matrix(y_true, y_pred, fold_idx, prefix=''):  
    """Draw Confusion Matrix"""  
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
    """Draw ROC curve"""  
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