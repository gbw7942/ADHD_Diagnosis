import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import DataLoader  
import pandas as pd  
import ast  
from tqdm import tqdm  
from data_loadervtwo import FMRIDataGenerator
from model import CNNLSTM

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

def train_one_epoch(model, train_loader, criterion, optimizer, device):  
    model.train()  
    running_loss = 0.0  
    correct = 0  
    total = 0  

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")  
    for _, batch in pbar:  
        X, y = batch  
        X, y = X.to(device), y.to(device)  

        outputs = model(X)  
        loss = criterion(outputs, y)  
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  

        running_loss += loss.item()  
        _, predicted = torch.max(outputs, 1)  
        correct += (predicted == y).sum().item()  
        total += y.size(0)  

        pbar.set_postfix({"Loss": loss.item(), "Accuracy": correct / total})  

    epoch_loss = running_loss / len(train_loader)  
    epoch_acc = correct / total  
    return epoch_loss, epoch_acc  


def validate(model, val_loader, criterion, device):  
    model.eval()  
    running_loss = 0.0  
    correct = 0  
    total = 0  


    pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating")  
    with torch.no_grad():  
        for _, batch in pbar:  
            X, y = batch  
            X, y = X.to(device), y.to(device)  

            outputs = model(X)  
            loss = criterion(outputs, y)  

            running_loss += loss.item()  
            _, predicted = torch.max(outputs, 1)  
            correct += (predicted == y).sum().item()  
            total += y.size(0)  
 
            pbar.set_postfix({"Loss": loss.item(), "Accuracy": correct / total})  

    epoch_loss = running_loss / len(val_loader)  
    epoch_acc = correct / total  
    return epoch_loss, epoch_acc  


def cross_validation_training(folds, preprocessed_dir, num_epochs=10, batch_size=32):  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    fold_results = []  

    for fold_idx, fold in enumerate(folds):  
        print(f"Starting Fold {fold_idx + 1}/{len(folds)}")  
 
        train_images = fold["train_images"]  
        train_labels = {img: label for img, label in zip(fold["train_images"], fold["train_labels"])}  
        val_images = fold["val_images"]  
        val_labels = {img: label for img, label in zip(fold["val_images"], fold["val_labels"])}  

        train_dataset = FMRIDataGenerator(train_images, train_labels, preprocessed_dir)  
        val_dataset = FMRIDataGenerator(val_images, val_labels, preprocessed_dir)  

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  

        model = CNNLSTM(num_classes=4, time_length=177, cnn_feature_dim=128).to(device)  
        criterion = nn.CrossEntropyLoss()  
        optimizer = optim.Adam(model.parameters(), lr=0.001)  

        # 训练和验证  
        for epoch in range(num_epochs):  
            print(f"\nFold {fold_idx + 1}, Epoch {epoch + 1}/{num_epochs}")  
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)  
            val_loss, val_acc = validate(model, val_loader, criterion, device)  

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")  
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")  

        fold_results.append({  
            "fold": fold_idx + 1,  
            "val_loss": val_loss,  
            "val_acc": val_acc  
        })  

    print("\nCross-Validation Results:")  
    for result in fold_results:  
        print(f"Fold {result['fold']}: Val Loss = {result['val_loss']:.4f}, Val Acc = {result['val_acc']:.4f}")  

    avg_val_acc = sum([result["val_acc"] for result in fold_results]) / len(fold_results)  
    print(f"\nAverage Validation Accuracy: {avg_val_acc:.4f}")  

if __name__ == "__main__":  
 
    csv_path = "/root/autodl-tmp/CNNLSTM/Project/fMRI/fold_data.csv"  
    preprocessed_dir = "/root/autodl-tmp/CNNLSTM/Project/preData"  
    folds = load_folds_from_csv(csv_path)  
 
    cross_validation_training(folds, preprocessed_dir, num_epochs=10, batch_size=4)