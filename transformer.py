import random
import math

import torch
import torch.nn as nn
from torch.nn.functional import softmax

class SelfAttention(nn.Module): # aka SingleHeadAttention to be combined into MultiHeadAttention
    def __init__(self, input_dim, d_k):
        super().__init__()
        self.Q_linear = nn.Linear(input_dim, d_k)
        self.K_linear = nn.Linear(input_dim, d_k)
        self.V_linear = nn.Linear(input_dim, d_k)

    def forward(self, Q, K , V, mask, dropout):
        Q_proj = self.Q_linear(Q)
        K_proj = self.K_linear(K)
        V_proj = self.V_linear(V)
        self.d_k = Q_proj[-1]

        attn_weights = torch.matmul(Q_proj, K_proj.transpose(0, 1)) / math.sqrt(self.d_k)
        attn_softmax = softmax(attn_weights, dim=-1) # on cols
        attn_values = torch.matmul(attn_softmax, V_proj)
        return attn_values 



d_model = 64
seq_len = 10
vocab_size = 100

d_k = 64


Q = torch.nn.Parameter(torch.rand(d_model, d_k))
K = torch.nn.Parameter(torch.rand(d_model, d_k))
V = torch.nn.Parameter(torch.rand(d_model, d_k))

self_attn = SelfAttention(d_model, d_k)
x = self_attn.forward()
print(x)