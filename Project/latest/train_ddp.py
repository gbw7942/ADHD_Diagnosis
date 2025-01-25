import torch  
import torch.nn as nn  
import torch.optim as optim  
import torch.distributed as dist  
from torch.nn.parallel import DistributedDataParallel as DDP  
from torch.utils.data import DataLoader  
from torch.utils.data.distributed import DistributedSampler  
from torch.cuda.amp import autocast, GradScaler  
from torch.optim.lr_scheduler import ReduceLROnPlateau  
import pandas as pd  
import numpy as np  
import ast  
from tqdm import tqdm  
import wandb  
import os  
import torch.multiprocessing as mp  
from sklearn.metrics import f1_score  
import config  
from data_loader import FMRIDataGenerator  
from model import ImprovedCNNTransformer  
from sklearn.model_selection import StratifiedKFold

def setup(rank, world_size):  
    """初始化分布式训练环境"""  
    os.environ['MASTER_ADDR'] = 'localhost'  
    os.environ['MASTER_PORT'] = '12355'  
    dist.init_process_group("nccl", rank=rank, world_size=world_size)  

def cleanup():  
    """清理分布式训练环境"""  
    dist.destroy_process_group()  

def load_folds_from_csv(csv_path):  
    if os.path.exists(csv_path):  
        print(f"Loading existing folds from {csv_path}")  
        df = pd.read_csv(csv_path)  
        folds = []  
        for _, row in df.iterrows():  
            fold_data = {  
                "train_images": ast.literal_eval(row["train_images"]),  
                "train_labels": ast.literal_eval(row["train_labels"]),  
                "val_images": ast.literal_eval(row["val_images"]),  
                "val_labels": ast.literal_eval(row["val_labels"]),  
            }  
            folds.append(fold_data)  
        return folds  
    
    else:  
        print(f"No existing folds found at {csv_path}. Creating new folds...")  
        metadata_csv = config.CSV_PATH + config.METADATA_CSV  
        file = pd.read_csv(metadata_csv)  
        
        list_IDs = file['Image'].tolist()  
        labels = dict(zip(file['Image'], file['DX']))  
        label_list = [1 if labels[img_id] != 0 else 0 for img_id in list_IDs]  
        
        skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=42)  
        fold_data = []  
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(list_IDs, label_list)):  
            fold_dict = {  
                'fold': fold + 1,  
                'train_images': [list_IDs[i] for i in train_idx],  
                'train_labels': [label_list[i] for i in train_idx],  
                'val_images': [list_IDs[i] for i in val_idx],  
                'val_labels': [label_list[i] for i in val_idx]  
            }  
            fold_data.append(fold_dict)  
        
        fold_df = pd.DataFrame(fold_data)  
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)  
        fold_df.to_csv(csv_path, index=False)  
        
        return fold_data  

def train_epoch(rank, model, train_loader, criterion, optimizer, scaler, device):  
    model.train()  
    running_loss = 0.0  
    predictions = []  
    true_labels = []  
    

    if rank == 0:  
        pbar = tqdm(total=len(train_loader), desc='Training',   
                   bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        

    for batch_idx, (data, target) in enumerate(train_loader):  
        data, target = data.to(device), target.to(device)  
        
        optimizer.zero_grad()  
        
        # 使用混合精度训练  
        with autocast():  
            output = model(data)  
            loss = criterion(output, target)  
        
        # 使用scaler进行反向传播  
        scaler.scale(loss).backward()  
        
        if config.GRAD_CLIP:  
            scaler.unscale_(optimizer)  
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)  
        
        scaler.step(optimizer)  
        scaler.update()  
        
        running_loss += loss.item()  
        _, predicted = torch.max(output.data, 1)  
        predictions.extend(predicted.cpu().numpy())  
        true_labels.extend(target.cpu().numpy())  

        # 更新进度条  
        if rank == 0:  
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})  
            pbar.update(1)  
    
    # 关闭进度条  
    if rank == 0:  
        pbar.close()

    # 收集所有进程的损失和指标  
    world_size = dist.get_world_size()  
    loss_tensor = torch.tensor(running_loss).to(device)  
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)  
    running_loss = loss_tensor.item() / world_size  

    # 收集所有进程的预测结果  
    all_predictions = [None for _ in range(world_size)]  
    all_labels = [None for _ in range(world_size)]  
    dist.all_gather_object(all_predictions, predictions)  
    dist.all_gather_object(all_labels, true_labels)  

    # 合并所有预测结果  
    predictions = np.concatenate(all_predictions)  
    true_labels = np.concatenate(all_labels)  

    epoch_loss = running_loss / len(train_loader)  
    epoch_acc = (np.array(predictions) == np.array(true_labels)).mean()  
    epoch_f1 = f1_score(true_labels, predictions, average='weighted')  
    dist.barrier()  
    return {  
        "loss": epoch_loss,  
        "accuracy": epoch_acc * 100,  
        "f1": epoch_f1  
    }  

def validate(rank, model, val_loader, criterion, device):  
    model.eval()  
    running_loss = 0.0  
    predictions = []  
    true_labels = []  


    if rank == 0:  
        pbar = tqdm(total=len(val_loader), desc='Validating',   
                   bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  
        


    with torch.no_grad():  
        for batch_idx, (data, target) in enumerate(val_loader):  
            data, target = data.to(device), target.to(device)  
            
            with autocast():  
                output = model(data)  
                loss = criterion(output, target)  
            
            running_loss += loss.item()  
            _, predicted = torch.max(output.data, 1)  
            predictions.extend(predicted.cpu().numpy())  
            true_labels.extend(target.cpu().numpy())  

            if rank == 0:  
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})  
                pbar.update(1) 

    
    # 关闭进度条  
    if rank == 0:  
        pbar.close()  

    # 收集所有进程的结果  
    world_size = dist.get_world_size()  
    loss_tensor = torch.tensor(running_loss).to(device)  
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)  
    running_loss = loss_tensor.item() / world_size  

    all_predictions = [None for _ in range(world_size)]  
    all_labels = [None for _ in range(world_size)]  
    dist.all_gather_object(all_predictions, predictions)  
    dist.all_gather_object(all_labels, true_labels)  

    predictions = np.concatenate(all_predictions)  
    true_labels = np.concatenate(all_labels)  
    dist.barrier()  
    return {  
        "loss": running_loss / len(val_loader),  
        "accuracy": (np.array(predictions) == np.array(true_labels)).mean() * 100,  
        "f1": f1_score(true_labels, predictions, average='weighted')  
    }  

def save_checkpoint(model, optimizer, fold_idx, epoch, metrics, checkpoint_dir):  
    """保存检查点"""  
    if dist.get_rank() == 0:  # 只在主进程保存  
        os.makedirs(checkpoint_dir, exist_ok=True)  
        checkpoint = {  
            'epoch': epoch + 1,  
            'model_state_dict': model.module.state_dict(),  # 注意这里用model.module  
            'optimizer_state_dict': optimizer.state_dict(),  
            'metrics': metrics  
        }  
        path = os.path.join(checkpoint_dir, f'checkpoint_fold_{fold_idx + 1}_epoch_{epoch + 1}.pth')  
        torch.save(checkpoint, path)  
        print(f"Checkpoint saved: {path}")  

def train_model(rank, world_size, folds, args):  
    """每个进程的训练函数"""  
    setup(rank, world_size)  
    device = torch.device(f"cuda:{rank}")  
    
    # 只在主进程初始化wandb  
    if rank == 0:  
        wandb.init(  
            project='fMRI',  
            config={  
                'num_classes': 2,  
                'learning_rate': args['learning_rate'],  
                'batch_size': args['batch_size'],  
                'num_epochs': args['num_epochs'],  
                'dropout_rate': config.DROP_OUT,  
                'weight_decay': config.WEIGHT_DECAY  
            }  
        )  

    for fold_idx, fold in enumerate(folds):  
        if rank == 0:  
            print(f"\nStarting Fold {fold_idx + 1}/{len(folds)}")  

        # 准备数据加载器  
        train_images = fold["train_images"]  
        train_labels = {img: label for img, label in zip(fold["train_images"], fold["train_labels"])}  
        val_images = fold["val_images"]  
        val_labels = {img: label for img, label in zip(fold["val_images"], fold["val_labels"])}  

        train_dataset = FMRIDataGenerator(train_images, train_labels, args['preprocessed_dir'])  
        val_dataset = FMRIDataGenerator(val_images, val_labels, args['preprocessed_dir'])  

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)  
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)  

        train_loader = DataLoader(  
            train_dataset,  
            batch_size=args['batch_size'],  
            sampler=train_sampler,  
            num_workers=10,  
            pin_memory=True  
        )  
        val_loader = DataLoader(  
            val_dataset,  
            batch_size=args['batch_size'],  
            sampler=val_sampler,  
            num_workers=10,  
            pin_memory=True  
        )  

        # 初始化模型和训练组件  
        model = ImprovedCNNTransformer().to(device)  
        model = DDP(model, device_ids=[rank])  
        
        criterion = nn.CrossEntropyLoss()  
        optimizer = optim.AdamW(  
            model.parameters(),  
            lr=args['learning_rate'],  
            weight_decay=config.WEIGHT_DECAY  
        )  
        scheduler = ReduceLROnPlateau(  
            optimizer,  
            mode='min',  
            factor=config.SCHEDULER_FACTOR,  
            patience=config.SCHEDULER_PATIENCE,  
            verbose=True  
        )  
        scaler = GradScaler()  

       
            
        best_val_acc = 0.0  
           # 添加epoch进度条  
        if rank == 0:  
            epoch_pbar = tqdm(total=args['num_epochs'], desc=f'Fold {fold_idx + 1}/{len(folds)}',  
                            position=0, leave=True)
        for epoch in range(args['num_epochs']):  
            train_sampler.set_epoch(epoch)  
            val_sampler.set_epoch(epoch)  

            # 训练和验证  
            train_metrics = train_epoch(rank, model, train_loader, criterion, optimizer, scaler, device)  
            val_metrics = validate(rank, model, val_loader, criterion, device)  

            # 学习率调度  
            scheduler.step(val_metrics['loss'])  

            # 在主进程上进行日志记录  
            if rank == 0:  
                # 更新epoch进度条  
                epoch_pbar.set_postfix({  
                    'Train Loss': f"{train_metrics['loss']:.4f}",  
                    'Val Loss': f"{val_metrics['loss']:.4f}",  
                    'Val Acc': f"{val_metrics['accuracy']:.2f}%"  
                })  
                epoch_pbar.update(1)  

                wandb.log({  
                    'train_loss': train_metrics['loss'],  
                    'train_accuracy': train_metrics['accuracy'],  
                    'train_f1': train_metrics['f1'],  
                    'val_loss': val_metrics['loss'],  
                    'val_accuracy': val_metrics['accuracy'],  
                    'val_f1': val_metrics['f1']  
                })  

                if val_metrics['accuracy'] > best_val_acc:  
                    best_val_acc = val_metrics['accuracy']  
                    save_checkpoint(  
                        model,   
                        optimizer,   
                        fold_idx,   
                        epoch,   
                        val_metrics,  
                        args['checkpoint_dir']  
                    )  

            dist.barrier()   

        if rank == 0:  
            epoch_pbar.close()  
            print(f"Fold {fold_idx + 1} completed. Best validation accuracy: {best_val_acc:.2f}%")

    if rank == 0:  
        wandb.finish()  
    
    cleanup()  

def main():  
    # 配置参数  
    args = {  
        'preprocessed_dir': config.PREPROCESSED_DIR,  
        'num_epochs': config.EPOCH,  
        'batch_size': config.BATCH_SIZE,  
        'learning_rate': config.LR,  
        'checkpoint_dir': '/root/autodl-tmp/CNNLSTM/Project/checkpoints/'  
    }  

    # 加载数据  
    csv_path = config.CSV_PATH + config.FOLD_DATA_CSV  
    folds = load_folds_from_csv(csv_path)  

    # 获取可用的GPU数量  
    world_size = torch.cuda.device_count()  
    print(f"Using {world_size} GPUs!")  

    # 修改这里的spawn调用  
    mp.spawn(  
        train_model,  
        args=(world_size, folds, args),  # 将kwargs作为普通参数传递  
        nprocs=world_size,  
        join=True  
    )  

if __name__ == "__main__":  
    main()