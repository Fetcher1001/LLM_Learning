import torch 
import torch.nn as nn 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout=0.1, qkv_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

        self.w_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_v = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)

    def forward(self, x):
      b, num_tokens, d_in = x.shape

      queries = self.w_q(x)
      keys = self.w_k(x)
      values = self.w_v(x)

      queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
      keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
      values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

      attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
      mask = self.mask[:num_tokens, :num_tokens].bool()
      attn_scores = attn_scores.masked_fill(mask, float())

      attn_weights = torch.softmax(attn_scores, dim=-1)
      attn_weights = self.dropout(attn_weights)
      attn_output = torch.matmul(attn_weights, values)

      attn_output = attn_output.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
      output = self.out_proj(attn_output)