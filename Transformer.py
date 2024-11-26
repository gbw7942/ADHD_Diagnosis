import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10): #input dimension and max len of input
        super().__init__()
        self.d_model=d_model
        pe = torch.zeros(max_len,d_model)
        pos = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1) #[[0],[pos_1],[pos_2]...[pos_n]]
        div_term = pos / pow(10000.0, torch.arange(0, d_model, 2).float() / d_model)
        pe[:,0::2] = torch.sin(div_term) #every two step (even)
        pe[:,1::2] = torch.cos(div_term)
        pe=pe.unsqueeze(0) #add num_batch dimension
        self.register_buffer('pe',pe)
    
    def forward(self,x):
        x = x * math.sqrt(self.d_model) 
        x = x + self.pe[:,:x.size(1)]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        assert d_model % heads == 0
        self.d_model=d_model
        self.heads=heads
        self.d_k= d_model//heads
        self.w_q = nn.Linear(d_model, d_model)  # wq * x = q
        self.w_k = nn.Linear(d_model, d_model)  # wk * x = k
        self.w_v = nn.Linear(d_model, d_model)  # wv * x = v
        self.dropout=nn.Dropout(p=dropout)
        self.out=nn.Linear(d_model,d_model)
        self.attention_scores = None  # Store attention scores for visualization


    def attention(self, q, k, v, d_k, mask):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k) #scaled dot product, [batch_size, heads, seq_len (len(q)), seq_len(len(k,v))] But in self attention both are seq_len of X
        print(f'scores:{scores.shape}')
        if mask is not None:
            scores = scores.masked_fill_(mask == 0, -1e9)
        scores = F.softmax(scores,dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        return output
    
    def forward(self,q, k, v, mask):
        bs = q.size(0)
        q = self.w_q(q).view(bs, -1, self.heads, self.d_k).transpose(1, 2) # convert from [bat_size, seq_len, d_model] to [batch_size, heads, seq_len, d_k]
        k = self.w_k(k).view(bs, -1, self.heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(bs, -1, self.heads, self.d_k).transpose(1, 2)
        print(f"q:{q.shape}")
        
        scores=self.attention(q,k,v,self.d_k,mask)
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

class MultiHeadAttentionWithVisualization(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        assert d_model % heads == 0
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(d_model, d_model)
        self.attention_scores = None  # Store attention scores for visualization

    def attention(self, q, k, v, d_k, mask):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill_(mask == 0, -1e9)
        
        # Store the attention scores before softmax for visualization
        self.attention_scores = scores.detach().cpu()
        
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        return output
    
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q = self.w_q(q).view(bs, -1, self.heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(bs, -1, self.heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(bs, -1, self.heads, self.d_k).transpose(1, 2)
        
        scores = self.attention(q, k, v, self.d_k, mask)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

def visualize_attention(attention_model, input_tensor, query_tensor=None):
    """
    Visualize attention scores across different heads.
    
    Args:
    - attention_model: MultiHeadAttentionWithVisualization instance
    - input_tensor: Input sequence tensor
    - query_tensor: Optional query tensor (if None, uses input_tensor)
    """
    if query_tensor is None:
        query_tensor = input_tensor
    
    # Perform forward pass to compute attention scores
    _ = attention_model(query_tensor, input_tensor, input_tensor)
    
    # Retrieve attention scores
    attention_scores = attention_model.attention_scores
    
    # Plot heatmaps for each head
    num_heads = attention_model.heads
    fig, axes = plt.subplots(1, num_heads, figsize=(5*num_heads, 5))
    
    for head in range(num_heads):
        # Select scores for the current head
        head_scores = attention_scores[0, head].numpy()
        
        # Plot heatmap
        if num_heads > 1:
            ax = axes[head]
        else:
            ax = axes
        
        sns.heatmap(head_scores, 
                    cmap='YlGnBu', 
                    ax=ax, 
                    cbar=True,
                    square=True,
                    xticklabels=range(head_scores.shape[1]),
                    yticklabels=range(head_scores.shape[0]))
        ax.set_title(f'Attention Scores - Head {head+1}')
        ax.set_xlabel('Key Sequence')
        ax.set_ylabel('Query Sequence')
    
    plt.tight_layout()
    plt.show()





def test():
    d_model=64
    seq_len=10
    batch_size=2
    heads=4  
    x=torch.randn(batch_size,seq_len,d_model)
    print(f'X: {x.shape}')
    # pe=PositionalEncoding(d_model=d_model)
    # result=pe.forward(x)
    # mha=MultiHeadAttention(d_model,heads)
    mha_vis = MultiHeadAttentionWithVisualization(d_model, heads)
    # result = mha_vis.forward(x,x,x,mask=None)
    # print(f'result:{result.shape}')
    visualize_attention(mha_vis,x,x)

test()
