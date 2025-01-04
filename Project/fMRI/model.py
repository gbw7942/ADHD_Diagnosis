import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vit_b_16

import config

class CNNLSTM(nn.Module):  
    def __init__(self, num_classes=config.NUM_CLASSES, time_length=config.TIME_LENGTH, cnn_feature_dim=config.FEATURE_DIM, dropout=None):  
        super(CNNLSTM, self).__init__()
        self.time_length = time_length  
        
        # CNN Module  
        self.cnn = nn.Sequential(  
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout3d(dropout),  # Added dropout after first conv layer
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout3d(dropout),  # Added dropout after second conv layer
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            nn.Conv3d(in_channels=64, out_channels=cnn_feature_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout3d(dropout),  # Added dropout after third conv layer
            nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        )  
        
        # LSTM Module  
        self.lstm = nn.LSTM(input_size=cnn_feature_dim, hidden_size=256, num_layers=2, batch_first=True, dropout=dropout)  
        
        # Dropout layer before final FC
        self.dropout = nn.Dropout(dropout)
        # Fully connected layer  
        self.fc = nn.Linear(256, num_classes) 

    def forward(self, x):  
        batch_size, self.time_length, _, _, _, _ = x.shape  
        x = x.view(batch_size * self.time_length, 1, 28, 28, 28)
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(batch_size, self.time_length, -1)
        lstm_out, _ = self.lstm(cnn_features)
        lstm_last_out = lstm_out[:, -1, :]
        # Apply dropout before final classification
        if self.dropout is not None:
            lstm_last_out = self.dropout(lstm_last_out)
        out = self.fc(lstm_last_out)
        return out

class CNNTransformer(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES, time_length=config.TIME_LENGTH, cnn_feature_dim=config.FEATURE_DIM, dropout=config.DROP_OUT):  
        super(CNNTransformer, self).__init__()
        self.time_length = time_length  
        
        # CNN Module with dropout
        self.cnn = nn.Sequential(  
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout3d(dropout),  # Added dropout
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout3d(dropout),  # Added dropout
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            nn.Conv3d(in_channels=64, out_channels=cnn_feature_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout3d(dropout),  # Added dropout
            nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        )  
        
        # Transformer Module (already has dropout in its implementation)
        self.transformer = nn.Transformer(
            d_model=cnn_feature_dim,
            nhead=config.NHEAD,
            num_encoder_layers=config.ENCODER_LAYERS,
            num_decoder_layers=config.DECODER_LAYERS,
            dim_feedforward=2048,
            dropout=dropout,
            activation='relu'
        )
        
        # Additional dropout before final FC
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(cnn_feature_dim, num_classes)

    def forward(self, x):
        batch_size, self.time_length, _, _, _, _ = x.shape  
        x = x.view(batch_size * self.time_length, 1, 28, 28, 28)
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(batch_size, self.time_length, -1)
        cnn_features = cnn_features.permute(1, 0, 2)
        transformer_out = self.transformer(cnn_features, cnn_features)
        last_step_output = transformer_out[-1, :, :]
        # Apply dropout before classification
        last_step_output = self.dropout(last_step_output)
        out = self.fc(last_step_output)
        return out

class ViT(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES, time_length=config.TIME_LENGTH, feature_dim=config.FEATURE_DIM, dropout=config.DROP_OUT):
        super(ViT, self).__init__()
        self.time_length = time_length
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.patch_embed = nn.Linear(time_length, feature_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, time_length, feature_dim))
        
        # Added dropout after embedding
        self.embed_dropout = nn.Dropout(dropout)

        # Encoder and Decoder (already have dropout in their implementation)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=config.NHEAD,
            dim_feedforward=2048,
            dropout=dropout,
            activation='relu'
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=config.ENCODER_LAYERS)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=config.NHEAD,
            dim_feedforward=2048,
            dropout=dropout,
            activation='relu'
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=config.DECODER_LAYERS)

        # Added dropout before final FC
        self.final_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        batch_size, time_length, feature_dim = x.shape
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x + self.pos_encoding[:, :feature_dim, :]
        # Apply dropout after embedding and positional encoding
        x = self.embed_dropout(x)
        x = x.permute(2, 0, 1)
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(x, encoder_out)
        output = decoder_out[-1, :, :]
        # Apply final dropout before classification
        output = self.final_dropout(output)
        out = self.fc(output)
        return out