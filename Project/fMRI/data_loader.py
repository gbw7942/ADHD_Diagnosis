import os  
import numpy as np  
import torch  
from torch.utils.data import Dataset, DataLoader  
from scipy.ndimage import zoom  
import nibabel as nib  
from tqdm import tqdm  
import ast 
import pandas as pd


def preprocess_and_save(dataset_dir, output_dir, list_IDs, time_length=177):  
    """  
    Preprocess fMRI data and save as .npy files.  
    Args:  
        dataset_dir (str): Directory containing raw .nii files.  
        output_dir (str): Directory to save preprocessed .npy files.  
        list_IDs (list): List of file names (IDs) to process.  
        time_length (int): Desired time length for fMRI data.  
    """  
    os.makedirs(output_dir, exist_ok=True)  
    for img_id in tqdm(list_IDs, desc="Preprocessing fMRI data"):  
        img_path = os.path.join(dataset_dir, img_id)  
        img = nib.load(img_path).get_fdata()  

        # Truncate or pad to the desired time length  
        if img.shape[3] > time_length:  
            img = img[:, :, :, :time_length]  
        elif img.shape[3] < time_length:  
            padding = np.zeros((img.shape[0], img.shape[1], img.shape[2], time_length - img.shape[3]))  
            img = np.concatenate((img, padding), axis=3)  

        # Resize each time point  
        scale_factors = (28 / img.shape[0], 28 / img.shape[1], 28 / img.shape[2])  
        resized_img = np.array([zoom(img[:, :, :, t], scale_factors, order=1) for t in range(time_length)])  
        resized_img = resized_img[..., np.newaxis]  # Add channel dimension  

        # Save as .npy  
        output_path = os.path.join(output_dir, f"{img_id}.npy")  
        np.save(output_path, resized_img)  


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
    从 CSV 文件加载 fold 数据。  
    Args:  
        csv_path (str): CSV 文件路径。  
    Returns:  
        folds (list): 包含每个 fold 数据的列表。  
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

def main():  
    raw_dataset_dir = "/root/autodl-tmp/CNNLSTM/Project/Data"  # 原始 .nii 文件目录  
    preprocessed_dir = "/root/autodl-tmp/CNNLSTM/Project/preData"  # 预处理后的 .npy 文件目录  
    csv_path = "/root/autodl-tmp/CNNLSTM/Project/fMRI/fold_data.csv"  # CSV 文件路径  

    folds = load_folds_from_csv(csv_path)  

    # all_images = set()  
    # for fold in folds:  
    #     all_images.update(fold["train_images"])  
    #     all_images.update(fold["val_images"])  
    # preprocess_and_save(raw_dataset_dir, preprocessed_dir, list(all_images))  

    # 处理示例：处理第一个 fold  
    fold = folds[0]  
    train_images = fold["train_images"]  
    train_labels = {img: label for img, label in zip(fold["train_images"], fold["train_labels"])}  
    val_images = fold["val_images"]  
    val_labels = {img: label for img, label in zip(fold["val_images"], fold["val_labels"])}  

    train_dataset = FMRIDataGenerator(train_images, train_labels, preprocessed_dir)  
    val_dataset = FMRIDataGenerator(val_images, val_labels, preprocessed_dir)  

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)  
 
    for batch in train_loader:  
        X, y = batch  
        print("训练数据形状：", X.shape, "训练标签：", y)  
        break  

    for batch in val_loader:  
        X, y = batch  
        print("验证数据形状：", X.shape, "验证标签：", y)  
        break  

if __name__ == "__main__":  
    main()  