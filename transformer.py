import random
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.functional import softmax, dropout

class SelfAttention(nn.Module): # aka SingleHeadAttention to be combined into MultiHeadAttention
    def __init__(self, input_dim, d_k):
        super().__init__()
        self.Q_linear = nn.Linear(input_dim, d_k)
        self.K_linear = nn.Linear(input_dim, d_k)
        self.V_linear = nn.Linear(input_dim, d_k)

    def forward(self, Q, K , V, attn_mask, attn_dropout): # adapted implementation of torch.scaled_dot_product_attention()
        Q_proj = self.Q_linear(Q)
        K_proj = self.K_linear(K)
        V_proj = self.V_linear(V)
        self.d_k = Q_proj.shape[0]
        attn_updated_mask = torch.zeros(size=(Q.shape[0], K.shape[0]))

        if attn_mask is not None:
            attn_mask = attn_mask.squeeze()
            attn_updated_mask.masked_fill_(attn_mask == 0, float("-inf")) # masked_fill_() > masked_fill()
            # attn_updated_mask = attn_updated_mask.masked_fill(attn_mask == 0, float("-inf"))

        attn_weights = torch.matmul(Q_proj, K_proj.transpose(0, 1)) / math.sqrt(self.d_k)
        print("1 -> \n", attn_weights)
        print("shape Q*K^T / sqrt(d_k) -> ", attn_weights.shape)
        print("mask applied -> \n", attn_updated_mask)
        attn_weights += attn_updated_mask
        print("2 -> \n", attn_weights) # will result in [token, -inf, -inf,...] for r1 and then [token, token, -inf,...] for r2 and so on
        attn_softmax = softmax(attn_weights, dim=-1) # on cols
        print("3 -> \n", attn_softmax)
        attn_softmax = dropout(attn_softmax, attn_dropout, training=True, inplace=True)
        print("4 -> \n", attn_softmax)
        attn_values = torch.matmul(attn_softmax, V_proj)
        print("5 -> \n", attn_values)
        return attn_values

class MultiHeadAttention(nn.Module):
    pass


d_k = 64 # overall size for the key and query vectors for single attn head in the model
d_model = 64 # overall size of the embedding dimension for the model
seq_len = 10
vocab_size = 100
batch_size = 1


Q = torch.nn.Parameter(torch.rand(d_model, d_k))
K = torch.nn.Parameter(torch.rand(d_model, d_k))
V = torch.nn.Parameter(torch.rand(d_model, d_k))

print("Q -> ", Q.shape)
print("K -> ", K.shape)
print("V -> ", V.shape)

self_attn = SelfAttention(d_model, d_k)

attn_mask = torch.tril(torch.ones(Q.shape[0], K.shape[0])).unsqueeze(0).repeat(batch_size, 1, 1)
print("attn_mask -> \n", attn_mask)
attn_dropout = 0.3

x = self_attn.forward(Q, K, V, attn_mask, attn_dropout)
print(x)
print("final shape ->", x.shape)