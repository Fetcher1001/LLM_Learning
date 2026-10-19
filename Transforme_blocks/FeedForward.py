import torch 
import torch.nn as nn
from GeLu import GeLuErf

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GeLuErf(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)



cfg = {"emb_dim": 768}
ff = FeedForward(cfg)
x = torch.randn(2, 10, 768)   # B=2, T=10, C=768
out = ff(x)
print(out.shape)              # muss (2, 10, 768) sein
assert out.shape == x.shape
print("Shape-Test bestanden")