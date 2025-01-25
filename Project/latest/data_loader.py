import os  
import numpy as np  
import torch  
from torch.utils.data import Dataset, DataLoader  
from scipy.ndimage import zoom  
import nibabel as nib  
from tqdm import tqdm  
import ast 
import pandas as pd
import random
import glob
from scipy.ndimage import rotate 

# class FMRIDataGenerator(Dataset):  
#     def __init__(self, list_IDs, labels, dataset_dir, transform=True, train=True):  
#         """  
#         Args:  
#             list_IDs (list): List of file names (IDs).  
#             labels (dict): Dictionary mapping file names to labels.  
#             dataset_dir (str): Directory containing preprocessed .npy files.  
#             transform (bool): Whether to apply data augmentation  
#             train (bool): Whether in training mode  
#         """  
#         self.dataset_dir = dataset_dir  
#         self.labels = labels  
#         self.list_IDs = list_IDs  
#         self.transform = transform  
#         self.train = train  

#     def random_flip(self, x):  
#         """随机翻转"""  
#         axes = [1, 2, 3]  # 空间维度的轴  
#         for axis in axes:  
#             if random.random() > 0.5:  
#                 x = np.flip(x, axis=axis)  
#         return x  

#     def random_rotate(self, x):  
#         """随机旋转(小角度)"""  
#         angle = random.uniform(-10, 10)  
#         axes = [(1,2), (2,3), (1,3)]  # 可能的旋转平面  
#         rot_plane = random.choice(axes)  
#         return rotate(x, angle, axes=rot_plane, reshape=False, mode='nearest')  

#     def add_gaussian_noise(self, x):  
#         """添加高斯噪声"""  
#         noise_level = random.uniform(0, 0.01)  
#         noise = np.random.normal(0, noise_level, x.shape)  
#         return x + noise  

#     def random_brightness_contrast(self, x):  
#         """随机调整亮度和对比度"""  
#         # 亮度调整  
#         brightness_factor = random.uniform(0.9, 1.1)  
#         x = x * brightness_factor  
        
#         # 对比度调整  
#         contrast_factor = random.uniform(0.9, 1.1)  
#         mean = np.mean(x)  
#         x = (x - mean) * contrast_factor + mean  
        
#         return x  

#     def apply_transforms(self, x):  
#         """应用数据增强"""  
#         if not self.transform or not self.train:  
#             return x  

#         # 随机应用不同的数据增强方法  
#         if random.random() > 0.5:  
#             x = self.random_flip(x)  
#         if random.random() > 0.5:  
#             x = self.random_rotate(x)  
#         if random.random() > 0.7:  
#             x = self.add_gaussian_noise(x)  
#         if random.random() > 0.7:  
#             x = self.random_brightness_contrast(x)  

#         return x  

#     def __len__(self):  
#         return len(self.list_IDs)  

#     def __getitem__(self, index):  
#         img_id = self.list_IDs[index]  
#         # Load preprocessed .npy file  
#         X = np.load(os.path.join(self.dataset_dir, f"{img_id}.npy"))  
        
#         # 应用数据增强  
#         X = self.apply_transforms(X)  
        
#         # 确保值域在合理范围内  
#         X = np.clip(X, 0, 1)  
        
#         y = self.labels[img_id]  
#         return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

class FMRIDataGenerator(Dataset):  
    def __init__(self, list_IDs, labels, dataset_dir):  
        """  
        Args:  
            list_IDs (list): List of file names (IDs).  
            labels (dict): Dictionary mapping file names to labels.  
            dataset_dir (str): Directory containing preprocessed .npy files.  
        """  
        self.dataset_dir = dataset_dir  
        self.labels = labels  
        self.list_IDs = list_IDs  

    def __len__(self):  
        return len(self.list_IDs)  

    def __getitem__(self, index):  
        img_id = self.list_IDs[index]  
        # Load preprocessed .npy file  
        X = np.load(os.path.join(self.dataset_dir, f"{img_id}.npy"))  
        y = self.labels[img_id]  
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)  


# 加载 CSV 文件并解析 fold 数据  
def load_folds_from_csv(csv_path):  
    """  
    Load fold data from a CSV file. 
    Args:  
        csv_path (str): CSV file path. 
    Returns:  
        folds (list): List of fold data.  
    """  
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

def test():
    # 超参数
    batch_size = 16
    raw_dataset_dir = "/root/autodl-tmp/CNNLSTM/Project/Data"  # Original .nii files directory  
    preprocessed_dir = "/root/autodl-tmp/CNNLSTM/Project/preprocssed_60_64"  # Preprocessed .npy files directory  
    csv_path = "/root/autodl-tmp/CNNLSTM/Project/fMRI/data.csv"  # CSV file containing fold data  


    folds = load_folds_from_csv(csv_path)  
    fold = folds[0]

    train_images = fold["train_images"]
    train_labels = {img: label for img, label in zip(fold["train_images"], fold["train_labels"])}
    val_images = fold["val_images"]
    val_labels = {img: label for img, label in zip(fold["val_images"], fold["val_labels"])}

    train_dataset = FMRIDataGenerator(train_images, train_labels, preprocessed_dir)
    val_dataset = FMRIDataGenerator(val_images, val_labels, preprocessed_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8,shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8,shuffle=False)
    # 获取第一个批次的数据并打印维度  
        
    for batch in train_loader:  
        X, y = batch  
        print("Train data shape: ", X.shape, "Train label: ", y)  
        break  

    for batch in val_loader:  
        X, y = batch  
        print("Val data shape: ", X.shape, "Val labe: ", y)  
        break  

if __name__ == "__main__":
    test()