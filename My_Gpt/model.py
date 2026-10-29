import torch 
import torch.nn as nn
from pathlib import Path
import json
from transformer_block import TransformerBlock

CONFIG_PATH = Path(__file__).with_name("config.json")
with CONFIG_PATH.open(encoding="utf-8") as config_file:
    GPT_CONFIG_124M = json.load(config_file)

class ShiGpt(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.position_embedding = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.out_head.weight = self.token_embedding.weight

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        token_emb = self.token_embedding(in_idx)
        pos_emb = self.position_embedding(torch.arange(seq_len, device=in_idx.device))
        x = token_emb + pos_emb
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits