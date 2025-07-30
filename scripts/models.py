import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=64, patch_size=8, channels=3, dim=128):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.dim = dim

        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x1 = self.projection(x)  # (B, dim, H', W')
        x2 = rearrange(x1, 'b c h w -> b (h w) c') # (B, num_patches, dim)
        return x2

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x
        
class ViT(nn.Module):
    def __init__(self, image_size=64, patch_size=8, channels=3, dim=128, depth=4, num_heads=4):
        super(ViT, self).__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, channels, dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.patch_embedding(x)  # (B, num_patches, dim)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)  # Final normalization
        return x
