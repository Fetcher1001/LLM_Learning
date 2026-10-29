import torch
import torch.nn as nn

from attention import MultiHeadAttention
from feed_forward import FeedForward
from layer_norm import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = MultiHeadAttention(cfg)
        self.feed_forward = FeedForward(cfg)
        self.layer_norm1 = LayerNorm(cfg["emb_dim"])
        self.layer_norm2 = LayerNorm(cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        x = x + self.drop(self.attention(self.layer_norm1(x)))
        x = x + self.drop(self.feed_forward(self.layer_norm2(x)))
        return x