import random
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.functional import softmax

class SelfAttention(nn.Module): # aka SingleHeadAttention to be combined into MultiHeadAttention
    def __init__(self, input_dim, d_k):
        super().__init__()
        self.Q_linear = nn.Linear(input_dim, d_k)
        self.K_linear = nn.Linear(input_dim, d_k)
        self.V_linear = nn.Linear(input_dim, d_k)

    def forward(self, Q, K , V, **kwargs): # mask: None, dropout: None):
        Q_proj = self.Q_linear(Q)
        K_proj = self.K_linear(K)
        V_proj = self.V_linear(V)
        self.d_k = Q_proj[-1].shape[0]

        attn_weights = torch.matmul(Q_proj, K_proj.transpose(0, 1)) / math.sqrt(self.d_k)
        print("Q*K^T / sqrt(d_k) -> ", attn_weights.shape)
        attn_softmax = softmax(attn_weights, dim=-1) # on cols
        attn_values = torch.matmul(attn_softmax, V_proj)
        return attn_values 


d_k = 64 # overall size for the key and query vectors for single attn head in the model
d_model = 64 # overall size of the embedding dimension for the model
seq_len = 10
vocab_size = 100


Q = torch.nn.Parameter(torch.rand(d_model, d_k))
K = torch.nn.Parameter(torch.rand(d_model, d_k))
V = torch.nn.Parameter(torch.rand(d_model, d_k))

print("Q -> ", Q.shape)
print("K -> ", K.shape)
print("V -> ", V.shape)

self_attn = SelfAttention(d_model, d_k)
x = self_attn.forward(Q, K, V)
print(x)
print("final shape ->", x.shape)