import torch 
import torch.nn as nn

class GeLuErf(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))
    
class GeLUApprox(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3)))) # GELU approximation using tanh correction_term =  0.044715 * torch.pow(x, 3)