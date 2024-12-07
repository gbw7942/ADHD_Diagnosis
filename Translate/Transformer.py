import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100): #input dimension and max len of input
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
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(d_model, d_model)

    def attention(self, q, k, v, d_k, mask=None):
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Handle mask broadcasting
        if mask is not None:
            # Ensure mask has the right number of dimensions
            while mask.dim() < scores.dim():
                mask = mask.unsqueeze(0)
            
            # If mask shape doesn't match scores, handle broadcasting
            if mask.shape != scores.shape:
                # Reshape mask to match the batch dimension of scores
                if mask.shape[-1] != scores.shape[-1]:
                    raise ValueError(f"Mask shape {mask.shape} is incompatible with scores shape {scores.shape}")
                
                # Broadcast mask to match scores shape
                mask = mask.expand(scores.shape[0], -1, -1, -1)
            
            # Apply mask by setting masked positions to large negative value
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        
        # Compute weighted sum
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
            bs = q.size(0)
            
            # Linear projections and reshape
            q = self.w_q(q).view(bs, -1, self.heads, self.d_k).transpose(1, 2)
            k = self.w_k(k).view(bs, -1, self.heads, self.d_k).transpose(1, 2)
            v = self.w_v(v).view(bs, -1, self.heads, self.d_k).transpose(1, 2)
            
            # Compute attention
            scores = self.attention(q, k, v, self.d_k, mask)
            
            # Concatenate and project
            concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
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

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn. Dropout(p=dropout)
    
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size=d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn, heads, dropout):
        super().__init__()
        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model,heads,dropout)
        self.ffn = PositionwiseFeedForward(d_model,d_ffn)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, enc_input, mask):
        enc_output = self.norm_1(enc_input + self.dropout_1(self.attn(enc_input, enc_input, enc_input, mask)))
        enc_output = self.norm_2(enc_output + self.dropout_2(self.ffn(enc_output)))
        return enc_output
    
class Encoder(nn.Module):
    def __init__(self, enc_vocab_size, d_model, d_ffn, N, heads, dropout):
        super().__init__()
        self.embed = nn.Embedding(enc_vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ffn, heads, dropout) for _ in range(N)])

    def forward(self, enc_input, enc_mask): # enc_input is [batch_size, seq_len]
        enc_out = self.embed(enc_input)
        enc_out = self.pe(enc_out)
        for layer in self.layers:
            enc_out = layer(enc_out, enc_mask)
        return enc_out    #[batch_size, seq_len, d_model]



class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn, heads, dropout = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, heads, dropout)
        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model)
        self.norm_3 = LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.dropout_3 = nn.Dropout(p=dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ffn)
    
    def forward(self, enc_out, enc_mask, dec_input, dec_mask):
        # Self-attention with masking
        residual = dec_input
        self_attn_output = self.self_attn(dec_input, dec_input, dec_input, dec_mask)
        dec_out = self.norm_1(residual + self.dropout_1(self_attn_output))

        # Cross-attention with encoder output
        residual = dec_out
        cross_attn_output = self.cross_attn(dec_out, enc_out, enc_out, enc_mask)
        dec_out = self.norm_2(residual + self.dropout_2(cross_attn_output))

        # Feed-forward
        residual = dec_out
        ffn_output = self.ffn(dec_out)
        dec_out = self.norm_3(residual + self.dropout_3(ffn_output))
        
        return dec_out
    
class Decoder(nn.Module):
    def __init__(self, dec_vocab_size, d_model, d_ffn, N, heads, dropout_prob):
        super().__init__()
        self.embed = nn.Embedding(dec_vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ffn, heads, dropout_prob) for _ in range(N)])

    def forward(self, enc_out, enc_mask, dec_input, dec_mask):
        # Embed and add positional encoding
        
        dec_out = self.embed(dec_input)
        dec_out = self.pe(dec_out)
        
        # Pass through decoder layers
        for layer in self.layers:
            dec_out = layer(enc_out, enc_mask, dec_out, dec_mask)
        
        return dec_out # [batch_size, seq_len,d_model]
    

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, d_ffn, N, heads, dropout):
        super().__init__()
        self.encode = Encoder(src_vocab_size, d_model, d_ffn, N, heads, dropout)
        self.decode = Decoder(tgt_vocab_size, d_model, d_ffn, N, heads, dropout)
        self.project = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, enc_input, enc_mask, dec_input, dec_mask):
        '''
        enc_input: [batch_size, src_len]
        dec_input: [batch_size, tgt_len]
        '''
        enc_out = self.encode(enc_input, enc_mask)
        dec_out = self.decode(enc_out, enc_mask, dec_input, dec_mask)
        project_out = self.project(dec_out)
        return project_out

    
def create_pad_mask(sequences, pad_idx=0):
    """
    创建padding mask
    sequences: [batch_size, seq_len]
    返回 [batch_size, 1, 1, seq_len] 的mask
    """
    batch_size, seq_len = sequences.size()
    mask = (sequences != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask

def create_subsequent_mask(size):
    """
    创建下三角mask，用于解码器的self-attention
    防止当前位置注意到后面的位置
    """
    mask = torch.triu(torch.ones((size, size)), diagonal=1).bool()
    return ~mask

def test(d_model=64, 
         seq_len=10, 
         batch_size=2, 
         heads=4, 
         n_layers=6, 
         d_ffn=2048, 
         dropout=0.1, 
         src_vocab_size=1000, 
         tgt_vocab_size=1000,
         src_sentences=None,
         tgt_sentences=None):
    """
    
    Args:
    - d_model: Model dimension
    - seq_len: Sequence length
    - batch_size: Batch size
    - heads: Number of attention heads
    - n_layers: Number of encoder/decoder layers
    - d_ffn: Dimension of feed-forward network
    - dropout: Dropout probability
    - src_vocab_size: Source vocabulary size
    - tgt_vocab_size: Target vocabulary size
    - src_sentences: Optional list of source sentences
    - tgt_sentences: Optional list of target sentences
    
    Returns:
    - Tuple of (encoder_output, decoder_output, transformer_output, predictions)
    """
    # Default sentences if not provided
    if src_sentences is None:
        src_sentences = [
            "machine learning is interesting",
            "I have time coding"
        ]
    
    if tgt_sentences is None:
        tgt_sentences = [
            "机器 学习 是 有趣",
            "我 有 时间 写代码"
        ]
    
    # Create dictionaries dynamically
    def create_vocab_dict(sentences):
        # Collect unique words
        words = set(word for sentence in sentences for word in sentence.split())
        # Create dictionary with special tokens
        vocab_dict = {
            "<PAD>": 0, 
            "<UNK>": 1, 
            **{word: idx+2 for idx, word in enumerate(words)}
        }
        return vocab_dict
    
    # Create vocabulary dictionaries
    src_dict = create_vocab_dict(src_sentences)
    tgt_dict = create_vocab_dict(tgt_sentences)
    
    # Process source language sentences
    def process_sentences(sentences, vocab_dict, max_len=None):
        if max_len is None:
            max_len = max(len(s.split()) for s in sentences)
        
        padded_sequences = []
        for sentence in sentences:
            tokens = [vocab_dict.get(word, vocab_dict["<UNK>"]) for word in sentence.split()]
            padding = [vocab_dict["<PAD>"]] * (max_len - len(tokens))
            padded_sequences.append(tokens + padding)
        
        return torch.tensor(padded_sequences)
    
    # Prepare input tensors
    enc_input = process_sentences(src_sentences, src_dict)
    dec_input = process_sentences(tgt_sentences, tgt_dict)
    
    # Create masks
    enc_mask = create_pad_mask(enc_input)
        
    # Decoder mask (padding + subsequent)
    size = dec_input.size(1)
    dec_pad_mask = create_pad_mask(dec_input)
    dec_subsequent_mask = create_subsequent_mask(size)
    dec_subsequent_mask = dec_subsequent_mask.unsqueeze(0).unsqueeze(0)
    dec_subsequent_mask = dec_subsequent_mask.expand(batch_size, 1, size, size)
    dec_mask = dec_pad_mask & dec_subsequent_mask
    
    # Initialize and test components
    encoder = Encoder(
        enc_vocab_size=len(src_dict), 
        d_model=d_model, 
        d_ffn=d_ffn, 
        N=n_layers, 
        heads=heads, 
        dropout=dropout
    )
    
    decoder = Decoder(
        dec_vocab_size=len(tgt_dict), 
        d_model=d_model, 
        d_ffn=d_ffn, 
        N=n_layers, 
        heads=heads, 
        dropout_prob=dropout
    )
    
    transformer = Transformer(
        src_vocab_size=len(src_dict), 
        tgt_vocab_size=len(tgt_dict), 
        d_model=d_model, 
        d_ffn=d_ffn, 
        N=n_layers, 
        heads=heads, 
        dropout=dropout
    )
    
    # Compute outputs
    enc_result = encoder(enc_input, enc_mask)
    dec_result = decoder(enc_result, enc_mask, dec_input, dec_mask)
    transformer_result = transformer(enc_input, enc_mask, dec_input, dec_mask)
    
    # Get predictions
    predictions = torch.argmax(transformer_result, dim=-1)
    
    # Print results and details
    print(f"\nInput Configuration:")
    print(f"Model Dimension: {d_model}")
    print(f"Sequence Length: {seq_len}")
    print(f"Batch Size: {batch_size}")
    print(f"Attention Heads: {heads}")
    print(f"Number of Layers: {n_layers}")
    
    print(f"\nOutput Shapes:")
    print(f"Encoder Output: {enc_result.shape}")
    print(f"Decoder Output: {dec_result.shape}")
    print(f"Transformer Output: {transformer_result.shape}")
    print(f"Predictions Shape: {predictions.shape}")
    
    print("\nTranslation Examples:")
    for i in range(len(src_sentences)):
        print(f"\nSource: {src_sentences[i]}")
        print(f"Target: {tgt_sentences[i]}")
        
        # Reverse lookup for predictions
        pred_indices = predictions[i].tolist()
        pred_words = [list(tgt_dict.keys())[list(tgt_dict.values()).index(idx)] for idx in pred_indices]
        print(f"Predicted: {' '.join(pred_words)}")
    
    return enc_result, dec_result, transformer_result, predictions

def main():
    # Default test
    print("\n--- Default Test ---")
    test()

if __name__ == "__main__":
    main()