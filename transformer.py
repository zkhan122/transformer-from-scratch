import random
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.functional import softmax, dropout

from attention import SelfAttention, MultiHeadAttention

class Encoder(nn.Module):
    def __init__(self, Q, K, V, input_dim, attn_embedding_dim, attn_dropout, num_heads, batch_size):
        super().__init__()
        self.Q = Q
        self.K = K
        self.V = V
        self.input_dim = input_dim
        self.attn_embedding_dim = attn_embedding_dim
        self.attn_dropout = attn_dropout
        self.num_heads = num_heads
        self.batch_size = batch_size


        self_attn = SelfAttention(self.input_dim, self.attn_embedding_dim)

        attn_mask = torch.tril(torch.ones(self.Q.shape[0], self.K.shape[0])).unsqueeze(0).repeat(self.batch_size, 1, 1)
        print("attn_mask -> \n", attn_mask)

        x = self_attn.forward(self.Q, self.K, self.V, attn_mask, self.attn_dropout)
        print(x)
        print("final dim of attn->", x.shape)
       
        print("\n\n------------------------")

        mha  = MultiHeadAttention(self.input_dim, x.shape[0], self.num_heads)
        print(f"\nMulti-Head-Attention initialized with (input_dim: {mha.input_dim}, attn_dim={mha.attn_embedding_dim}, {mha.num_heads} heads")
        multi_head_attn = mha.forward(self.Q, self.K, self.V, attn_mask, self.attn_dropout)

        print(multi_head_attn)
        print(multi_head_attn.shape)





d_k = 64 # overall size for the key and query vectors for single attn head in the model
d_model = 64 # overall size of the embedding dimension for the model for Q, K, V (esp because K, V are passed to the decoder)
seq_len = 10
vocab_size = 100
batch_size = 1
n_heads = 3
attn_dim = 64
attn_dropout = 0.3

Q = torch.nn.Parameter(torch.rand(d_model, d_k))
K = torch.nn.Parameter(torch.rand(d_model, d_k))
V = torch.nn.Parameter(torch.rand(d_model, d_k))

print("Q -> ", Q.shape)
print("K -> ", K.shape)
print("V -> ", V.shape)


encoder = Encoder(Q, K, V, d_model, d_k,  attn_dropout, n_heads, batch_size)
