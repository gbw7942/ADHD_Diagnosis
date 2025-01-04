import warnings
warnings.filterwarnings("ignore")

from data_loader import FMRIDataGenerator

import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL

# ============================ DATA WORK ============================

file_num = sys.argv[1]

# Dataframes
dataset_dir = "/root/autodl-tmp/CNNLSTM/Project/Data"
model_train_data = pd.read_csv("/root/autodl-tmp/CNNLSTM/Project/fMRI/data/training_data_{}".format(file_num))
model_val_data = pd.read_csv("/root/autodl-tmp/CNNLSTM/Project/fMRI/data/validatation_data_{}".format(file_num))

# Dictionary of data values
partition = {'train': model_train_data['Image'].values,
             'validation': model_val_data['Image'].values}

# Training Data
train_labels = {row['Image']: row['DX'] for _, row in model_train_data.iterrows()}
val_labels = {row['Image']: row['DX'] for _, row in model_val_data.iterrows()}

batch_size = 6
train_dataset = FMRIDataGenerator(partition['train'], train_labels, dataset_dir, batch_size)
val_dataset = FMRIDataGenerator(partition['validation'], val_labels, dataset_dir, batch_size)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ============================ MODEL ARCHITECTURE ============================

# class CNN_LSTM_Model(nn.Module):
#     def __init__(self):
#         super(CNN_LSTM_Model, self).__init__()
        
#         self.conv3d = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3)
#         self.relu = nn.ReLU()
#         self.pool3d = nn.MaxPool3d(kernel_size=2, stride=2)
#         self.flatten = nn.Flatten()
        
#         self.lstm = nn.LSTM(input_size=64 * 13 * 13 * 13, hidden_size=10, batch_first=True, dropout=0.3)
#         self.fc = nn.Linear(10, 1)
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):
#         batch_size, time_steps, C, H, W, D = x.size()
#         x = x.view(batch_size * time_steps, C, H, W, D)  # Merge batch and time dimensions
#         x = self.conv3d(x)
#         x = self.relu(x)
#         x = self.pool3d(x)
#         x = self.flatten(x)
#         x = x.view(batch_size, time_steps, -1)  # Restore batch and time dimensions
        
#         x, _ = self.lstm(x)
#         x = self.fc(x[:, -1, :])  # Take output from the last time step
#         x = self.sigmoid(x)
#         return x.squeeze()

class CNN_LSTM_Model(nn.Module):  
    def __init__(self):  
        super(CNN_LSTM_Model, self).__init__()  
        
        # 3D CNN部分  
        self.conv3d = nn.Sequential(  
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1),  
            nn.ReLU(),  
            nn.MaxPool3d(kernel_size=2, stride=2),  
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  
            nn.ReLU(),  
            nn.MaxPool3d(kernel_size=2, stride=2)  
        )  
        
        # 计算CNN输出大小  
        self.cnn_output_size = 64 * 7 * 7 * 7  
        
        # LSTM部分  
        self.lstm = nn.LSTM(  
            input_size=self.cnn_output_size,  
            hidden_size=256,  
            num_layers=2,  
            batch_first=True,  
            dropout=0.5  
        )  
        
        # 全连接层  
        self.fc = nn.Sequential(  
            nn.Linear(256, 64),  
            nn.ReLU(),  
            nn.Dropout(0.5),  
            nn.Linear(64, 1),  
            nn.Sigmoid()  
        )  
    
    def forward(self, x):  
        # 打印输入张量的形状  
        print(f"Input shape: {x.shape}")  
        
        # 确保输入维度正确  
        if len(x.shape) == 6:  # (batch, time, channel, x, y, z)  
            batch_size, time_steps = x.shape[0], x.shape[1]  
            x = x.view(batch_size * time_steps, 1, 28, 28, 28)  
        elif len(x.shape) == 5:  # (batch, time, x, y, z)  
            batch_size, time_steps = x.shape[0], x.shape[1]  
            x = x.view(batch_size * time_steps, 1, 28, 28, 28)  
        
        # 打印重塑后的张量形状  
        print(f"Reshaped for CNN: {x.shape}")  
        
        # CNN处理  
        x = self.conv3d(x)  
        print(f"After CNN: {x.shape}")  
        
        # 展平CNN输出  
        x = x.view(-1, self.cnn_output_size)  
        print(f"After flatten: {x.shape}")  
        
        # 重组为序列  
        x = x.view(batch_size, time_steps, self.cnn_output_size)  
        print(f"Reshaped for LSTM: {x.shape}")  
        
        # LSTM处理  
        x, _ = self.lstm(x)  
        print(f"After LSTM: {x.shape}")  
        
        # 只使用最后一个时间步的输出  
        x = self.fc(x[:, -1, :])  
        print(f"Final output: {x.shape}")  
        
        return x.squeeze()  

# Model Initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_LSTM_Model().to(device)

# Optimizer and Loss
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.BCELoss()

# ============================ TRAINING ============================

epochs = 5
logger_path = "/root/autodl-tmp/CNNLSTM/Project/log/adhd-fmri-history_cv{num}_{time}.csv".format(
    num=file_num, time=f'{datetime.now():%H-%M-%S%z_%m%d%Y}'
)

train_steps_per_epoch = len(train_dataset)
validate_steps_per_epoch = len(val_dataset)

# Training Loop
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for images, labels in train_dataset:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_dataset:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/train_steps_per_epoch:.4f}, "
          f"Val Loss: {val_loss/validate_steps_per_epoch:.4f}")
    
    # Logging
    with open(logger_path, 'a') as log_file:
        log_file.write(f"{epoch+1},{train_loss/train_steps_per_epoch:.4f},{val_loss/validate_steps_per_epoch:.4f}\n")
