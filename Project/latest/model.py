import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.optim.lr_scheduler import ReduceLROnPlateau  
import numpy as np  
from tqdm import tqdm  
import torch.nn.functional as F
import math
from einops.layers.torch import Rearrange

class CNNLSTM(nn.Module):  
    def __init__(self):  
        super(CNNLSTM, self).__init__()  
        
        # 3D CNN部分  
        self.conv1 = nn.Sequential(  
            nn.Conv3d(1, 8, kernel_size=3, padding=1),  
            nn.BatchNorm3d(8),  
            nn.ReLU(),  
            nn.MaxPool3d(2)  
        )  
        
        self.conv2 = nn.Sequential(  
            nn.Conv3d(8, 16, kernel_size=3, padding=1),  
            nn.BatchNorm3d(16),  
            nn.ReLU(),  
            nn.MaxPool3d(2)  
        )  
        
        self.conv3 = nn.Sequential(  
            nn.Conv3d(16, 32, kernel_size=3, padding=1),  
            nn.BatchNorm3d(32),  
            nn.ReLU(),  
            nn.MaxPool3d(2)  
        )  
        
        # LSTM部分  
        self.lstm = nn.LSTM(  
            input_size=32 * 8 * 8 * 8,  # CNN输出展平后的特征维度  
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
            nn.Linear(64, 2)  # 二分类问题  
        )  

    def forward(self, x):  
        # 输入 x 形状: [batch, time_steps, H, W, D, channel]  
        batch_size, time_steps = x.size(0), x.size(1)  
        
        # 重塑输入以处理每个时间步  
        x = x.view(batch_size * time_steps, 1, 64, 64, 64)  
        
        # CNN特征提取  
        x = self.conv1(x)  
        x = self.conv2(x)  
        x = self.conv3(x)  
        
        # 展平CNN输出  
        x = x.view(batch_size, time_steps, -1)  
        
        # LSTM处理时序特征  
        lstm_out, _ = self.lstm(x)  
        
        # 只使用最后一个时间步的输出  
        x = lstm_out[:, -1, :]  
        
        # 全连接层分类  
        x = self.fc(x)  
        return x  


class ResBlock3D(nn.Module):  
    def __init__(self, in_channels, out_channels):  
        super(ResBlock3D, self).__init__()  
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)  
        self.bn1 = nn.BatchNorm3d(out_channels)  
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)  
        self.bn2 = nn.BatchNorm3d(out_channels)  
        
        self.shortcut = nn.Sequential()  
        if in_channels != out_channels:  
            self.shortcut = nn.Sequential(  
                nn.Conv3d(in_channels, out_channels, kernel_size=1),  
                nn.BatchNorm3d(out_channels)  
            )  
    
    def forward(self, x):  
        residual = x  
        x = F.relu(self.bn1(self.conv1(x)))  
        x = self.bn2(self.conv2(x))  
        x += self.shortcut(residual)  
        x = F.relu(x)  
        return x  

class ImprovedCNNLSTM(nn.Module):  
    def __init__(self):  
        super(ImprovedCNNLSTM, self).__init__()  
        
        # Initial conv  
        self.conv1 = nn.Sequential(  
            nn.Conv3d(1, 32, kernel_size=7, stride=2, padding=3),  
            nn.BatchNorm3d(32),  
            nn.ReLU(),  
            nn.MaxPool3d(2)  
        )  
        
        # ResBlocks  
        self.res1 = ResBlock3D(32, 64)  
        self.res2 = ResBlock3D(64, 128)  
        self.res3 = ResBlock3D(128, 256)  
        
        # Global Average Pooling  
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))  
        
        # LSTM  
        self.lstm = nn.LSTM(  
            input_size=256,  
            hidden_size=128,  
            num_layers=1,  
            batch_first=True,  
            bidirectional=True  
        )  
        
        # Attention  
        self.attention = nn.Sequential(  
            nn.Linear(256, 64),  
            nn.Tanh(),  
            nn.Linear(64, 1),  
            nn.Softmax(dim=1)  
        )  
        
        # Classification  
        self.classifier = nn.Sequential(  
            nn.Linear(256, 64),  
            nn.LayerNorm(64),  
            nn.ReLU(),  
            nn.Dropout(0.5),  
            nn.Linear(64, 2)  
        )  
        
        # 初始化权重  
        self.apply(self._init_weights)  
    
    def _init_weights(self, m):  
        if isinstance(m, nn.Conv3d):  
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
        elif isinstance(m, nn.BatchNorm3d):  
            nn.init.constant_(m.weight, 1)  
            nn.init.constant_(m.bias, 0)  
    
    def forward(self, x):  
        batch_size, time_steps = x.size(0), x.size(1)  
        x = x.view(batch_size * time_steps, 1, 64, 64, 64)  
        
        # CNN feature extraction  
        x = self.conv1(x)  
        x = self.res1(x)  
        x = self.res2(x)  
        x = self.res3(x)  
        
        # Global average pooling  
        x = self.gap(x)  
        x = x.view(batch_size, time_steps, -1)  
        
        # Bidirectional LSTM  
        lstm_out, _ = self.lstm(x)  
        
        # Attention  
        attn_weights = self.attention(lstm_out)  
        context = torch.sum(attn_weights * lstm_out, dim=1)  
        
        # Classification  
        output = self.classifier(context)  
        return output  

class PositionalEncoding(nn.Module):  
    def __init__(self, d_model, max_len=15):  
        super().__init__()  
        position = torch.arange(max_len).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  
        pe = torch.zeros(max_len, 1, d_model)  
        pe[:, 0, 0::2] = torch.sin(position * div_term)  
        pe[:, 0, 1::2] = torch.cos(position * div_term)  
        self.register_buffer('pe', pe)  

    def forward(self, x):  
        """  
        Args:  
            x: Tensor, shape [seq_len, batch_size, embedding_dim]  
        """  
        return x + self.pe[:x.size(0)]  
    
class ImprovedCNNTransformer(nn.Module):  
    def __init__(self, num_classes=2):  
        super(ImprovedCNNTransformer, self).__init__()  
        
        # CNN backbone  
        self.conv1 = nn.Sequential(  
            nn.Conv3d(1, 32, kernel_size=7, stride=2, padding=3),  
            nn.BatchNorm3d(32),  
            nn.ReLU(),  
            nn.MaxPool3d(2)  
        )  
        
        # ResBlocks  
        self.res1 = ResBlock3D(32, 64)  
        self.res2 = ResBlock3D(64, 128)  
        self.res3 = ResBlock3D(128, 256)  
        
        # Global Average Pooling  
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))  
        
        # Transformer parameters  
        self.d_model = 256  
        self.nhead = 8  
        self.num_layers = 2  
        
        # Positional Encoding  
        self.pos_encoder = PositionalEncoding(self.d_model)  
        
        # Transformer Encoder  
        encoder_layer = nn.TransformerEncoderLayer(  
            d_model=self.d_model,  
            nhead=self.nhead,  
            dim_feedforward=1024,  
            dropout=0.1,  
            activation='gelu',  
            batch_first=True  
        )  
        self.transformer_encoder = nn.TransformerEncoder(  
            encoder_layer,  
            num_layers=self.num_layers  
        )  
        
        # Layer Normalization  
        self.norm = nn.LayerNorm(self.d_model)  
        
        # Classification head  
        self.classifier = nn.Sequential(  
            nn.Linear(self.d_model, 64),  
            nn.LayerNorm(64),  
            nn.GELU(),  
            nn.Dropout(0.5),  
            nn.Linear(64, num_classes)  
        )  
        
        # Initialize weights  
        self.apply(self._init_weights)  
    
    def _init_weights(self, m):  
        if isinstance(m, nn.Conv3d):  
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
        elif isinstance(m, nn.BatchNorm3d):  
            nn.init.constant_(m.weight, 1)  
            nn.init.constant_(m.bias, 0)  
        elif isinstance(m, nn.Linear):  
            nn.init.xavier_uniform_(m.weight)  
            if m.bias is not None:  
                nn.init.constant_(m.bias, 0)  
    
    def forward(self, x):  
        # x shape: [batch_size, time_steps, 1, 64, 64, 64]  
        batch_size, time_steps = x.size(0), x.size(1)  
        # x = x.view(batch_size * time_steps, 1, 128,128,128)  
        x = x.view(batch_size * time_steps, 1, 64,64,64)  
        # CNN feature extraction  
        x = self.conv1(x)  
        x = self.res1(x)  
        x = self.res2(x)  
        x = self.res3(x)  
        
        # Global average pooling  
        x = self.gap(x)  
        x = x.view(batch_size, time_steps, -1)  # [batch_size, time_steps, d_model]  
        
        # Add positional encoding  
        x = x * math.sqrt(self.d_model)  # Scale embeddings  
        
        # Transformer encoding  
        x = self.transformer_encoder(x)  
        
        # Global pooling over sequence length  
        x = torch.mean(x, dim=1)  # [batch_size, d_model]  
        
        # Layer normalization  
        x = self.norm(x)  
        
        # Classification  
        output = self.classifier(x)  
        
        return output  

    def get_attention_weights(self, x):  
        """  
        Get attention weights for visualization  
        """  
        batch_size, time_steps = x.size(0), x.size(1)  
        # x = x.view(batch_size * time_steps, 1, 128,128,128)  
        x = x.view(batch_size * time_steps, 1, 64,64,64)  
        # CNN feature extraction  
        x = self.conv1(x)  
        x = self.res1(x)  
        x = self.res2(x)  
        x = self.res3(x)  
        
        # Global average pooling  
        x = self.gap(x)  
        x = x.view(batch_size, time_steps, -1)  
        
        # Scale embeddings  
        x = x * math.sqrt(self.d_model)  
        
        # Get attention weights from the first layer  
        attention_weights = []  
        for layer in self.transformer_encoder.layers:  
            # Forward pass through self-attention  
            attention_weights.append(  
                layer.self_attn(x, x, x)[1]  # Get attention weights  
            )  
        
        return attention_weights  



class VitTransformer(nn.Module):  
    def __init__(self, num_classes=2, patch_size=4, in_channels=1, dim=256, depth=6, heads=8, mlp_dim=512, dropout=0.1):  
        super(VitTransformer, self).__init__()  
        
        # Image dimensions (assuming 64x64x64 input)  
        self.image_size = 64  
        self.patch_size = patch_size  
        self.num_patches = (self.image_size // patch_size) ** 3  
        self.patch_dim = in_channels * patch_size ** 3  
        self.dim = dim  
        self.num_classes = num_classes  
        
        # Patch embedding - 修改Rearrange以适应新的输入格式  
        self.patch_embedding = nn.Sequential(  
            # 首先调整channels的位置  
            Rearrange('b t h w d c -> b t c h w d'),  
            # 然后进行patch划分  
            Rearrange('b t c (h p1) (w p2) (d p3) -> b t (h w d) (p1 p2 p3 c)',  
                     p1=patch_size, p2=patch_size, p3=patch_size),  
            nn.LayerNorm(self.patch_dim),  
            nn.Linear(self.patch_dim, dim),  
            nn.LayerNorm(dim),  
        )  
        
        # Class token and position embedding  
        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, dim))  
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, self.num_patches + 1, dim))  
        
        # Dropout  
        self.dropout = nn.Dropout(dropout)  
        
        # Transformer encoder  
        encoder_layer = nn.TransformerEncoderLayer(  
            d_model=dim,  
            nhead=heads,  
            dim_feedforward=mlp_dim,  
            dropout=dropout,  
            activation='gelu',  
            batch_first=True,  
            norm_first=True  
        )  
        self.transformer_encoder = nn.TransformerEncoder(  
            encoder_layer,  
            num_layers=depth  
        )  
        
        # Layer normalization  
        self.norm = nn.LayerNorm(dim)  
        
        # Classification head  
        self.classifier = nn.Sequential(  
            nn.LayerNorm(dim),  
            nn.Linear(dim, mlp_dim),  
            nn.GELU(),  
            nn.Dropout(dropout),  
            nn.Linear(mlp_dim, num_classes)  
        )  
        
        # Initialize weights  
        self.apply(self._init_weights)  
        
    def _init_weights(self, m):  
        if isinstance(m, nn.Linear):  
            torch.nn.init.xavier_uniform_(m.weight)  
            if m.bias is not None:  
                nn.init.constant_(m.bias, 0)  
        elif isinstance(m, nn.LayerNorm):  
            nn.init.constant_(m.weight, 1.0)  
            nn.init.constant_(m.bias, 0)  
        elif isinstance(m, nn.Parameter):  
            nn.init.normal_(m, std=0.02)  
            
    def forward(self, x):  
        # x shape: [batch_size, time_steps, height, width, depth, channels]  
        batch_size, time_steps = x.size(0), x.size(1)  
        
        # Patch embedding  
        x = self.patch_embedding(x)  # [batch_size, time_steps, num_patches, dim]  
        
        # Add class token to each time step  
        cls_tokens = self.cls_token.expand(batch_size, time_steps, 1, self.dim)  
        x = torch.cat([cls_tokens, x], dim=2)  # [batch_size, time_steps, num_patches + 1, dim]  
        
        # Add position embedding  
        x = x + self.pos_embedding  
        
        # Apply dropout  
        x = self.dropout(x)  
        
        # Reshape for transformer input  
        x = x.view(batch_size * time_steps, self.num_patches + 1, self.dim)  
        
        # Transformer encoding  
        x = self.transformer_encoder(x)  
        
        # Reshape back  
        x = x.view(batch_size, time_steps, self.num_patches + 1, self.dim)  
        
        # Use only the class token output  
        x = x[:, :, 0]  # [batch_size, time_steps, dim]  
        
        # Average over time steps  
        x = torch.mean(x, dim=1)  # [batch_size, dim]  
        
        # Layer normalization  
        x = self.norm(x)  
        
        # Classification  
        output = self.classifier(x)  
        
        return output  

    def get_attention_weights(self, x):  
        """  
        Get attention weights for visualization  
        """  
        batch_size, time_steps = x.size(0), x.size(1)  
        
        # Patch embedding  
        x = self.patch_embedding(x)  
        
        # Add class token  
        cls_tokens = self.cls_token.expand(batch_size, time_steps, 1, self.dim)  
        x = torch.cat([cls_tokens, x], dim=2)  
        
        # Add position embedding  
        x = x + self.pos_embedding  
        
        # Apply dropout  
        x = self.dropout(x)  
        
        # Reshape for transformer  
        x = x.view(batch_size * time_steps, self.num_patches + 1, self.dim)  
        
        # Get attention weights from all layers  
        attention_weights = []  
        for layer in self.transformer_encoder.layers:  
            attention_weights.append(  
                layer.self_attn(x, x, x)[1]  # Get attention weights  
            )  
        
        return attention_weights
    

def test():  
    # 创建模型实例  
    # model = CNNLSTM()  
    model = ImprovedCNNLSTM()
    # model = ImprovedCNNTransformer() 
    # model = VitTransformer()  
    
    # 将模型移至可用设备  
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  
    model = model.to(device)  
    print(f"Using device: {device}")  
    
    # 创建随机测试数据  
    batch_size = 8  
    time_steps = 60  
    channels = 1  
    height = width = depth = 64  
    
    # 创建随机输入数据和标签  
    # 修正输入数据的形状为 [batch_size, time_steps, channels, height, width, depth]  
    dummy_input = torch.randn(batch_size, time_steps,height, width, depth, channels).to(device)  
    dummy_labels = torch.randint(0, 2, (batch_size,)).to(device)  
    
    print("\nInput shape:", dummy_input.shape)  
    print("Labels:", dummy_labels)  
    
    # 测试前向传播  
    print("\nTesting forward pass...")  
    try:  
        with torch.no_grad():  
            output = model(dummy_input)  
            print("Output shape:", output.shape)  
            print("Output values:", output)  
            
            # 计算预测结果  
            _, predicted = torch.max(output.data, 1)  
            print("\nPredicted classes:", predicted)  
            
        print("\nModel architecture:")  
        print(model)  
        
        # 计算模型参数总数  
        total_params = sum(p.numel() for p in model.parameters())  
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  
        print(f"\nTotal parameters: {total_params:,}")  
        print(f"Trainable parameters: {trainable_params:,}")  
        
        print("\nForward pass test completed successfully!")  
        
    except Exception as e:  
        print(f"Error during forward pass: {str(e)}")  
        import traceback  
        traceback.print_exc()  
        
    # 测试损失函数  
    print("\nTesting loss calculation...")  
    try:  
        criterion = nn.CrossEntropyLoss()  
        loss = criterion(output, dummy_labels)  
        print(f"Test loss value: {loss.item():.4f}")  
        
    except Exception as e:  
        print(f"Error during loss calculation: {str(e)}")
        
    # 测试优化器  
    # print("\nTesting optimizer...")  
    # try:  
    #     optimizer = optim.Adam(model.parameters(), lr=0.001)  
    #     loss.backward()  
    #     optimizer.step()  
    #     print("Optimizer test completed successfully!")  
        
    # except Exception as e:  
    #     print(f"Error during optimization: {str(e)}")  

if __name__ == "__main__":  
    test()  


