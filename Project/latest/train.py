import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import DataLoader  
from torch.optim.lr_scheduler import ReduceLROnPlateau  
import numpy as np  
from tqdm import tqdm  
import os  
from datetime import datetime  
import json  
from data_loader import load_folds_from_csv,FMRIDataGenerator
from model import *
from sklearn.metrics import f1_score, roc_curve, auc, precision_recall_curve 
# import config
import random
import wandb



class Trainer:  
    def __init__(self, model, train_loader, val_loader, config, num_classes=2):  
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
        self.model = model.to(self.device)  
        self.train_loader = train_loader  
        self.val_loader = val_loader  
        self.config = config
        self.num_classes=num_classes
        # 初始化优化器和损失函数  
        self.criterion = nn.CrossEntropyLoss()  
        self.optimizer = optim.AdamW(  # 使用AdamW优化器  
            self.model.parameters(),   
            lr=config['learning_rate'],  
            weight_decay=config['weight_decay'],  
            betas=(0.9, 0.999)  
        )  
        self.scheduler = ReduceLROnPlateau(  
            self.optimizer,   
            mode='min',   
            factor=0.5,   
            patience=5,   
            verbose=True  
        )  
        
        # 创建保存目录  
        self.save_dir = os.path.join(  
            'experiments',   
            datetime.now().strftime("%Y%m%d_%H%M%S")  
        )  
        os.makedirs(self.save_dir, exist_ok=True)  
        
        # 保存配置  
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:  
            json.dump(config, f, indent=4)  
        
        # 初始化最佳指标  
        self.best_val_loss = float('inf')  
        self.best_val_acc = 0.0  
        
        # 训练记录  
        self.train_losses = []  
        self.val_losses = []  
        self.train_accs = []  
        self.val_accs = []  
        
        # 添加类别标签  
        self.class_names = [f"Class_{i}" for i in range(num_classes)]  
        
        # 初始化wandb  
        self.run = wandb.init(  
            project='fMRI',  
            config={  
                'model_type': model.__class__.__name__,  
                'num_classes': num_classes,  
                'learning_rate': config['learning_rate'],  
                'batch_size': config['batch_size'],  
                'num_epochs': config['epochs'],  
                'weight_decay': config['weight_decay'],  
                'grad_clip': config['grad_clip'],  
                'scheduler_patience': config['scheduler_patience'],  
                'scheduler_factor': config['scheduler_factor']  
            }  
        )  
        
        # 创建wandb Table列定义  
        self.columns = ["epoch", "split"] + [f"pred_{i}" for i in range(num_classes)] + ["target"]  
        
        # Watch model  
        wandb.watch(self.model, log="all", log_freq=100)  
        
    def compute_metrics(self, outputs, targets):  
        """扩展的metrics计算函数"""  
        predictions = outputs.cpu().numpy()  
        targets = targets.cpu().numpy()  
        pred_classes = np.argmax(predictions, axis=1)  
        
        # 转换为one-hot编码  
        targets_one_hot = np.eye(self.num_classes)[targets]  
        
        # 基础指标  
        metrics = {  
            'f1_micro': f1_score(targets, pred_classes, average='micro'),  
            'f1_macro': f1_score(targets, pred_classes, average='macro'),  
            'f1_weighted': f1_score(targets, pred_classes, average='weighted')  
        }  
        
        # 每个类别的F1分数  
        f1_per_class = f1_score(targets, pred_classes, average=None)  
        for i, f1 in enumerate(f1_per_class):  
            metrics[f'f1_class_{i}'] = f1  
        
        # ROC和AUC  
        fpr = dict()  
        tpr = dict()  
        roc_auc = dict()  
        
        for i in range(self.num_classes):  
            fpr[i], tpr[i], _ = roc_curve(targets_one_hot[:, i], predictions[:, i])  
            roc_auc[i] = auc(fpr[i], tpr[i])  
            metrics[f'auc_class_{i}'] = roc_auc[i]  
        
        # PR曲线数据  
        precision = dict()  
        recall = dict()  
        pr_auc = dict()  
        
        for i in range(self.num_classes):  
            precision[i], recall[i], _ = precision_recall_curve(  
                targets_one_hot[:, i],   
                predictions[:, i]  
            )  
            pr_auc[i] = auc(recall[i], precision[i])  
            metrics[f'pr_auc_class_{i}'] = pr_auc[i]  
        
        return {  
            **metrics,  
            'predictions': predictions,  
            'pred_classes': pred_classes,  
            'targets': targets,  
            'fpr': fpr,  
            'tpr': tpr,  
            'precision': precision,  
            'recall': recall  
        }  

    def log_metrics_to_wandb(self, epoch, split, metrics):  
        """记录详细的metrics到wandb"""  
        # 基础指标  
        log_dict = {  
            f'{split}/loss': metrics['loss'],  
            f'{split}/accuracy': metrics['accuracy'],  
            f'{split}/f1_micro': metrics['f1_micro'],  
            f'{split}/f1_macro': metrics['f1_macro'],  
            f'{split}/f1_weighted': metrics['f1_weighted'],  
            'epoch': epoch  
        }  

        # 每个类别的指标  
        for i in range(self.num_classes):  
            log_dict.update({  
                f'{split}/f1_class_{i}': metrics[f'f1_class_{i}'],  
                f'{split}/auc_class_{i}': metrics[f'auc_class_{i}'],  
                f'{split}/pr_auc_class_{i}': metrics[f'pr_auc_class_{i}']  
            })  

        # 混淆矩阵  
        wandb.log({  
            f'{split}/confusion_matrix': wandb.plot.confusion_matrix(  
                probs=None,  
                y_true=metrics['targets'],  
                preds=metrics['pred_classes'],  
                class_names=self.class_names  
            )  
        })  

        # ROC曲线  
        for i in range(self.num_classes):  
            # 创建ROC曲线数据表  
            roc_table = wandb.Table(  
                columns=["fpr", "tpr", "threshold"],  
                data=[[x, y, 0] for x, y in zip(  
                    metrics['fpr'][i].tolist(),  
                    metrics['tpr'][i].tolist()  
                )]  
            )  
            wandb.log({  
                f'{split}/roc_class_{i}': wandb.plot.line(  
                    roc_table,  
                    "fpr",  
                    "tpr",  
                    title=f'{split} ROC Curve for {self.class_names[i]}'  
                )  
            })  

        # PR曲线  
        for i in range(self.num_classes):  
            # 创建PR曲线数据表  
            pr_table = wandb.Table(  
                columns=["precision", "recall", "threshold"],  
                data=[[x, y, 0] for x, y in zip(  
                    metrics['precision'][i].tolist(),  
                    metrics['recall'][i].tolist()  
                )]  
            )  
            wandb.log({  
                f'{split}/pr_curve_class_{i}': wandb.plot.line(  
                    pr_table,  
                    "recall",  
                    "precision",  
                    title=f'{split} PR Curve for {self.class_names[i]}'  
                )  
            })  

        # 预测分布  
        pred_table = wandb.Table(  
            columns=["predictions"],  
            data=[[p] for p in metrics['predictions'].flatten()]  
        )  
        wandb.log({  
            f'{split}/pred_dist': wandb.plot.histogram(  
                pred_table,  
                "predictions",  
                title=f'{split} Prediction Distribution'  
            )  
        })  

        # 记录所有指标  
        wandb.log(log_dict) 
    
    def _save_history(self, train_metrics, val_metrics):  
        """保存训练历史到类属性中"""  
        # 保存损失  
        self.train_losses.append(train_metrics['loss'])  
        self.val_losses.append(val_metrics['loss'])  
        
        # 保存准确率  
        self.train_accs.append(train_metrics['accuracy'])  
        self.val_accs.append(val_metrics['accuracy'])  
        
        # 保存到文件  
        history = {  
            'train_loss': self.train_losses,  
            'val_loss': self.val_losses,  
            'train_acc': self.train_accs,  
            'val_acc': self.val_accs  
        }  
        
        # 保存为JSON文件  
        with open(os.path.join(self.save_dir, 'training_history.json'), 'w') as f:  
            json.dump(history, f, indent=4)  
        
        # 保存详细metrics  
        detailed_metrics = {  
            'epoch': len(self.train_losses) - 1,  
            'train': train_metrics,  
            'val': val_metrics  
        }  
        
        # 将numpy数组转换为列表以便JSON序列化  
        def convert_to_serializable(obj):  
            if isinstance(obj, np.ndarray):  
                return obj.tolist()  
            elif isinstance(obj, dict):  
                return {k: convert_to_serializable(v) for k, v in obj.items()}  
            elif isinstance(obj, (list, tuple)):  
                return [convert_to_serializable(x) for x in obj]  
            return obj  
        
        detailed_metrics = convert_to_serializable(detailed_metrics)  
        
        # 保存详细metrics  
        metrics_file = os.path.join(self.save_dir, f'metrics_epoch_{len(self.train_losses)}.json')  
        with open(metrics_file, 'w') as f:  
            json.dump(detailed_metrics, f, indent=4)

    def train_epoch(self, epoch):  
        self.model.train()  
        total_loss = 0  
        correct = 0  
        total = 0  
        
        # 存储所有batch的输出和标签用于计算metrics  
        all_outputs = []  
        all_targets = []  
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [Train]')  
        for batch_idx, (data, target) in enumerate(pbar):  
            data, target = data.to(self.device), target.to(self.device)  
            # 前向传播  
            self.optimizer.zero_grad()  
            output = self.model(data)  
            loss = self.criterion(output, target)  
            
            # 反向传播  
            loss.backward()  
            
            # 梯度裁剪  
            if self.config['grad_clip']:  
                torch.nn.utils.clip_grad_norm_(  
                    self.model.parameters(),  
                    self.config['grad_clip']  
                )  
            
            self.optimizer.step()  
            # 计算准确率  
            _, predicted = output.max(1)  
            total += target.size(0)  
            correct += predicted.eq(target).sum().item()  
            
            # 存储输出和标签用于计算metrics  
            all_outputs.append(output.detach().cpu())  
            all_targets.append(target.cpu())  
            # 更新总损失  
            total_loss += loss.item()  
            
            # 更新进度条  
            pbar.set_postfix({  
                'loss': f'{total_loss/(batch_idx+1):.4f}',  
                'acc': f'{100.*correct/total:.2f}%'  
            })  
            
            # Log batch metrics to wandb  
            wandb.log({  
                'batch': epoch * len(self.train_loader) + batch_idx,  
                'batch/train_loss': loss.item(),  
                'batch/train_accuracy': 100. * correct / total,  
                'learning_rate': self.optimizer.param_groups[0]['lr']  
            })  
        
        # 连接所有batch的输出和标签  
        all_outputs = torch.cat(all_outputs, dim=0)  
        all_targets = torch.cat(all_targets, dim=0)  
        
        # 计算epoch级别的metrics  
        epoch_metrics = {  
            'loss': total_loss / len(self.train_loader),  
            'accuracy': correct / total,  
            **self.compute_metrics(all_outputs, all_targets)  
        }  
        
        # 记录训练metrics  
        self.log_metrics_to_wandb(epoch, 'train', epoch_metrics)  
        
        return epoch_metrics  

    def validate(self, epoch):  
        self.model.eval()  
        total_loss = 0  
        correct = 0  
        total = 0  
        
        # 存储所有batch的输出和标签用于计算metrics  
        all_outputs = []  
        all_targets = []  
        all_predicts = []
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [Val]')  
        
        with torch.no_grad():  
            for batch_idx, (data, target) in enumerate(pbar):  
                data, target = data.to(self.device), target.to(self.device)  
                
                # 前向传播  
                output = self.model(data)  
                loss = self.criterion(output, target)  
                
                # 计算准确率  
                _, predicted = output.max(1)  
                total += target.size(0)  
                correct += predicted.eq(target).sum().item()  
                
                # 存储输出和标签用于计算metrics  
                all_outputs.append(output.cpu())  
                all_targets.append(target.cpu())  
                all_predicts.append(predicted.cpu())
                # 更新总损失  
                total_loss += loss.item()  
                
                # 更新进度条  
                pbar.set_postfix({  
                    'loss': f'{total_loss/(batch_idx+1):.4f}',  
                    'acc': f'{100.*correct/total:.2f}%'  
                })  
                
                # Log batch metrics to wandb  
                wandb.log({  
                    'batch': epoch * len(self.val_loader) + batch_idx,  
                    'batch/val_loss': loss.item(),  
                    'batch/val_accuracy': 100. * correct / total  
                })  
        
        # 连接所有batch的输出和标签 
        all_outputs = torch.cat(all_outputs, dim=0)  
        all_targets = torch.cat(all_targets, dim=0)  
        print(all_targets, all_predicts)
        # 计算epoch级别的metrics  
        epoch_metrics = {  
            'loss': total_loss / len(self.val_loader),  
            'accuracy': correct / total,  
            **self.compute_metrics(all_outputs, all_targets)  
        }  
        
        # 记录验证metrics  
        self.log_metrics_to_wandb(epoch, 'val', epoch_metrics)  
        
        return epoch_metrics 

    def save_checkpoint(self, epoch, val_loss, val_acc, is_best=False):  
        checkpoint = {  
            'epoch': epoch,  
            'model_state_dict': self.model.state_dict(),  
            'optimizer_state_dict': self.optimizer.state_dict(),  
            'val_loss': val_loss,  
            'val_acc': val_acc,  
            'config': self.config  
        }  
        
        # 保存当前检查点  
        torch.save(  
            checkpoint,  
            os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')  
        )  
        
        # 如果是最佳模型，额外保存一份  
        if is_best:  
            torch.save(  
                checkpoint,  
                os.path.join(self.save_dir, 'best_model.pth')  
            )  

    def train(self):  
        print(f"Training on device: {self.device}")  
        print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")  
        
        for epoch in range(self.config['epochs']):  
            # 训练  
            train_metrics = self.train_epoch(epoch)  
            
            # 验证  
            val_metrics = self.validate(epoch)  
            
            # 更新学习率  
            self.scheduler.step(val_metrics['loss'])  
            
            # 检查是否是最佳模型  
            is_best = False  
            if val_metrics['accuracy'] > self.best_val_acc:  
                self.best_val_acc = val_metrics['accuracy']  
                is_best = True  
            if val_metrics['loss'] < self.best_val_loss:  
                self.best_val_loss = val_metrics['loss']  
            
            # 保存检查点  
            self.save_checkpoint(  
                epoch,   
                val_metrics['loss'],   
                val_metrics['accuracy'],   
                is_best  
            )  
            
            # 打印epoch总结  
            self._print_epoch_summary(epoch, train_metrics, val_metrics)  
            
            # 保存训练历史  
            self._save_history(train_metrics, val_metrics)  
        
        # 结束wandb运行  
        wandb.finish()  

    def _print_epoch_summary(self, epoch, train_metrics, val_metrics):  
        """打印每个epoch的详细总结"""  
        print(f'\nEpoch {epoch+1}/{self.config["epochs"]}:')  
        print(f'Train Loss: {train_metrics["loss"]:.4f}, Train Acc: {train_metrics["accuracy"]*100:.2f}%')  
        print(f'Train F1 (micro/macro/weighted): {train_metrics["f1_micro"]:.4f}/{train_metrics["f1_macro"]:.4f}/{train_metrics["f1_weighted"]:.4f}')  
        print(f'Val Loss: {val_metrics["loss"]:.4f}, Val Acc: {val_metrics["accuracy"]*100:.2f}%')  
        print(f'Val F1 (micro/macro/weighted): {val_metrics["f1_micro"]:.4f}/{val_metrics["f1_macro"]:.4f}/{val_metrics["f1_weighted"]:.4f}')  
        print(f'Best Val Acc: {self.best_val_acc*100:.2f}%')  
        print(f'Best Val Loss: {self.best_val_loss:.4f}\n') 

def main():  
    # 训练配置  
    config = {  
    'learning_rate': 0.0003,  # 降低学习率  
    'weight_decay': 1e-5,     # 减小权重衰减  
    'epochs': 200,  
    'grad_clip': 0.5,         # 减小梯度裁剪阈值  
    'batch_size': 6,          # 增加批次大小  
    'scheduler_patience': 10,  # 增加学习率调度器的耐心值  
    'scheduler_factor': 0.1    # 更激进的学习率衰减  
    }  

    
    # 准备数据加载器  
    preprocessed_dir = "/root/autodl-tmp/CNNLSTM/Project/preprocssed_60_64"  
    csv_path = "/root/autodl-tmp/CNNLSTM/Project/fMRI/fold_data.csv"  
    
    folds = load_folds_from_csv(csv_path)  
    fold = folds[0]
    selected_indices = random.sample(range(len(fold["train_images"])), 205)

    # 准备数据加载器  
    train_images = fold["train_images"]  
    train_labels = {img: label for img, label in zip(fold["train_images"], fold["train_labels"])}  
    val_images = [fold["train_images"][i] for i in selected_indices]
    val_labels = {img: fold["train_labels"][i] for i, img in enumerate(train_images)}

    train_dataset = FMRIDataGenerator(train_images, train_labels, preprocessed_dir)  
    val_dataset = FMRIDataGenerator(val_images, val_labels, preprocessed_dir)  
    
    train_loader = DataLoader(  
        train_dataset,   
        batch_size=config['batch_size'],   
        shuffle=True,   
        num_workers=8 
    )  
    val_loader = DataLoader(  
        val_dataset,   
        batch_size=config['batch_size'],   
        shuffle=False,   
        num_workers=8 
    )  
    
    # 初始化模型  
    # model = ImprovedCNNLSTM()  # 或者使用原始的CNNLSTM  
    model = ImprovedCNNTransformer()
    
    # 创建训练器并开始训练  
    trainer = Trainer(model, train_loader, val_loader, config)  
    trainer.train()  

if __name__ == "__main__":  
    main()