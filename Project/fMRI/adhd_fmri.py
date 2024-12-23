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

import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# ============================ DATA WORK ============================

file_num = sys.argv[1]

# Dataframes
dataset_dir = "/pylon5/cc5614p/deopha32/fmri_images/model_data/"
model_train_data = pd.read_csv("/home/deopha32/ADHD-FMRI/Data/training_data_{}".format(file_num))
model_val_data = pd.read_csv("/home/deopha32/ADHD-FMRI/Data/validatation_data_{}".format(file_num))

# Dictionary of data values
partition = {'train': model_train_data['Image'].values,
             'validation': model_val_data['Image'].values}

# Training Data
train_labels = {row['Image']: row['DX'] for _, row in model_train_data.iterrows()}
val_labels = {row['Image']: row['DX'] for _, row in model_val_data.iterrows()}

# PyTorch Dataset Class
class FMRI_Dataset(Dataset):
    def __init__(self, image_list, labels, dataset_dir):
        self.image_list = image_list
        self.labels = labels
        self.dataset_dir = dataset_dir
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.dataset_dir, self.image_list[idx])
        image = np.load(image_path)  # Assuming image is stored as .npy
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dim
        label = torch.tensor(self.labels[self.image_list[idx]], dtype=torch.float32)
        return image, label

# Loaders
batch_size = 6
train_dataset = FMRI_Dataset(partition['train'], train_labels, dataset_dir)
val_dataset = FMRI_Dataset(partition['validation'], val_labels, dataset_dir)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ============================ MODEL ARCHITECTURE ============================

class CNN_LSTM_Model(nn.Module):
    def __init__(self):
        super(CNN_LSTM_Model, self).__init__()
        
        self.conv3d = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3, activation='relu')
        self.pool3d = nn.MaxPool3d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        
        self.lstm = nn.LSTM(input_size=64 * 13 * 13 * 13, hidden_size=10, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size, time_steps, C, H, W, D = x.size()
        x = x.view(batch_size * time_steps, C, H, W, D)  # Merge batch and time dimensions
        x = self.conv3d(x)
        x = self.pool3d(x)
        x = self.flatten(x)
        x = x.view(batch_size, time_steps, -1)  # Restore batch and time dimensions
        
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Take output from the last time step
        x = self.sigmoid(x)
        return x.squeeze()

# Model Initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_LSTM_Model().to(device)

# Optimizer and Loss
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.BCELoss()

# ============================ TRAINING ============================

epochs = 500
logger_path = "/pylon5/cc5614p/deopha32/Saved_Models/adhd-fmri-history_cv{num}_{time}.csv".format(
    num=file_num, time=f'{datetime.now():%H-%M-%S%z_%m%d%Y}'
)

train_steps_per_epoch = len(train_loader)
validate_steps_per_epoch = len(val_loader)

# Training Loop
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
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
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/train_steps_per_epoch:.4f}, "
          f"Val Loss: {val_loss/validate_steps_per_epoch:.4f}")
    
    # Logging
    with open(logger_path, 'a') as log_file:
        log_file.write(f"{epoch+1},{train_loss/train_steps_per_epoch:.4f},{val_loss/validate_steps_per_epoch:.4f}\n")
