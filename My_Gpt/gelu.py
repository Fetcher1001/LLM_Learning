import torch
import torch.nn as nn

class Gelu(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))