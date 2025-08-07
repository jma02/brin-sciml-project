import torch 
import torch.nn as nn
import torch.nn.functional as F

from modules import AttentionBlock1D, FourierBlock1D

class FourierAttn(nn.Module):
    def __init__(self, input_dim=1, output_dim=2, hidden_dim=128, num_heads=8, num_layers=3, dropout=0.1):
        super(FourierAttn, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Fourier-Attention blocks
        self.blocks = nn.ModuleList([
            FourierAttnBlock(hidden_dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Positional encoding for time
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)
        
    def forward(self, t):
        # Ensure t is the right shape: (batch_size, seq_len, input_dim)
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # Add feature dimension
        if t.dim() == 2 and t.size(-1) != self.input_dim:
            t = t.unsqueeze(0)  # Add batch dimension
            
        # Input projection
        x = self.input_proj(t)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply Fourier-Attention blocks
        for block in self.blocks:
            x = block(x)
        
        # Output projection
        output = self.output_proj(x)
        
        return output

class FourierAttnBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = AttentionBlock1D(dim, num_heads)
        self.fourier = FourierBlock1D(dim)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Attention with residual connection
        x = x + self.attention(self.norm1(x))
        
        # Fourier transform with residual connection
        x = x + self.fourier(self.norm2(x))
        
        # Feed-forward with residual connection
        x = x + self.ffn(self.norm3(x))
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)