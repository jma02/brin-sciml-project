import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import torch.fft

class AttentionBlock1D(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "Embedding dim must be divisible by num_heads"

        self.norm = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: Tensor of shape (B, L, D)
        """
        x_norm = self.norm(x)

        # Linear projections
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        # Reshape for multi-head: (B, L, H, D/H) → (B, H, L, D/H)
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)

        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(q, k, v)  # shape: (B, H, L, D/H)

        # Back to (B, L, D)
        out = rearrange(out, 'b h l d -> b l (h d)')

        # Residual connection with final projection
        return (x + self.proj_out(out)) / np.sqrt(2.0)


class FourierBlock1D(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        # Linear projection in Fourier space (real + imag as separate features)
        self.freq_proj = nn.Linear(dim * 2, dim * 2)  # Keep the same size for proper reshaping

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: Tensor of shape (B, L, D)
        """
        x_norm = self.norm(x)
        B, L, D = x_norm.shape

        # Compute Fourier transform along the sequence (L) dimension
        x_freq = torch.fft.rfft(x_norm, dim=1)  # shape (B, L//2+1, D), complex

        # Split into real and imaginary parts for real-valued ops
        x_freq_real = torch.view_as_real(x_freq)  # shape (B, Lf, D, 2)
        x_freq_real = x_freq_real.reshape(B, x_freq.shape[1], -1)  # (B, Lf, 2*D)

        # Apply linear projection in Fourier space
        x_freq_proj = self.freq_proj(x_freq_real)  # (B, Lf, 2*D)

        # Convert back to complex representation
        x_freq_proj = x_freq_proj.reshape(B, x_freq.shape[1], D, 2)  # (B, Lf, D, 2)
        x_freq_proj = torch.view_as_complex(x_freq_proj.contiguous())  # (B, Lf, D)

        # Inverse FFT back to time domain
        x_out = torch.fft.irfft(x_freq_proj, n=L, dim=1)  # shape (B, L, D)

        # Final projection + residual
        x_out = self.out_proj(self.dropout(x_out))

        return (x + x_out) / np.sqrt(2.0)