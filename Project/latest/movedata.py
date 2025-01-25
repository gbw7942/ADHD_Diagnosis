import pandas as pd  
import ast  
import os  
import shutil  
from pathlib import Path  

def load_folds_and_move_unused(csv_path, data_dir, backup_dir):  
    """  
    Load fold data from CSV and move unused files to backup directory.  
    
    Args:  
        csv_path (str): CSV file path  
        data_dir (str): Original data directory path  
        backup_dir (str): Backup directory path for unused files  
    Returns:  
        folds (list): List of fold data  
        moved_files (list): List of moved files  
    """  
    # 创建备份目录（如果不存在）  
    os.makedirs(backup_dir, exist_ok=True)  
    
    # 读取CSV文件  
    df = pd.read_csv(csv_path)  
    folds = []  
    all_files_in_csv = set()  
    
    # 收集CSV中的所有文件  
    for _, row in df.iterrows():  
        fold_data = {  
            "train_images": ast.literal_eval(row["train_images"]),  
            "train_labels": ast.literal_eval(row["train_labels"]),  
            "val_images": ast.literal_eval(row["val_images"]),  
            "val_labels": ast.literal_eval(row["val_labels"]),  
        }  
        folds.append(fold_data)  
        
        # 将所有文件名添加到集合中  
        all_files_in_csv.update(fold_data["train_images"])  
        all_files_in_csv.update(fold_data["train_labels"])  
        all_files_in_csv.update(fold_data["val_images"])  
        all_files_in_csv.update(fold_data["val_labels"])  
    
    # 获取数据目录中的所有文件  
    existing_files = set(f for f in os.listdir(data_dir) if f.endswith('.nii.gz'))  
    
    # 找出需要移动的文件（在数据目录中存在但不在CSV中的文件）  
    files_to_move = existing_files - all_files_in_csv  
    moved_files = []  
    
    # 移动文件  
    for file_name in files_to_move:  
        src_path = os.path.join(data_dir, file_name)  
        dst_path = os.path.join(backup_dir, file_name)  
        try:  
            shutil.move(src_path, dst_path)  
            moved_files.append(file_name)  
            print(f"Moved: {file_name}")  
        except Exception as e:  
            print(f"Error moving {file_name}: {str(e)}")  
    
    print(f"\nTotal files moved: {len(moved_files)}")  
    return folds, moved_files  

# 使用示例：  
csv_path = "/root/autodl-tmp/CNNLSTM/Project/fMRI/fold_data.csv"  
data_dir = "/root/autodl-tmp/CNNLSTM/Project/Data"  
backup_dir = "/root/autodl-tmp/CNNLSTM/Project/backupData"  
folds, moved_files = load_folds_and_move_unused(csv_path, data_dir, backup_dir)  