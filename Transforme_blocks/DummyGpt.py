import json
from pathlib import Path

import torch
import torch.nn as nn


CONFIG_PATH = Path(__file__).with_name("config.json")
with CONFIG_PATH.open(encoding="utf-8") as config_file:
    GPT_CONFIG_124M = json.load(config_file)

class DummyGpt(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.position_embedding = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[DummyTrfBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self,in_idx):
        batch_size, seq_len = in_idx.shape
        token_emb = self.token_embedding(in_idx)
        pos_emb = self.position_embedding(torch.arange(seq_len, device=in_idx.device))
        x = token_emb + pos_emb
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class DummyTrfBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()

    def forward(self,x):
        return x

class DummyLayerNorm(nn.Module):
    def __init__(self,normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self,x):
        return self.layer_norm(x)


import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode_ordinary(txt1)))
batch.append(torch.tensor(tokenizer.encode_ordinary(txt2)))

batch = torch.stack(batch, dim=0)

torch.manual_seed(123)
model = DummyGpt(GPT_CONFIG_124M)
logits = model(batch)