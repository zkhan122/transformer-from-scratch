import random
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.functional import softmax, dropout

from attention import SelfAttention, MultiHeadAttention
from embedding import PositionalEmbeddingLayer

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
        ff_hidden_dim=None


        if ff_hidden_dim is None:
            ff_hidden_dim = 4 * input_dim

        self.ff_hidden_dim = ff_hidden_dim


        self.feed_forward = None
        

        self_attn = SelfAttention(self.input_dim, self.attn_embedding_dim)

        self.attn_mask = torch.tril(torch.ones(self.Q.shape[0], self.K.shape[0])).unsqueeze(0).repeat(self.batch_size, 1, 1)
        print("attn_mask -> \n", self.attn_mask)

        self.x = self_attn.forward(self.Q, self.K, self.V, self.attn_mask, self.attn_dropout)
        print(self.x)
        print("final dim of attn->", self.x.shape)
       
        print("\n\n------------------------")

        self.mha  = MultiHeadAttention(self.input_dim, self.x.shape[0], self.num_heads)

    
    def forward(self):
        
        multi_head_attn = self.mha.forward(self.Q, self.K, self.V, self.attn_mask, self.attn_dropout)
        print(f"\nMulti-Head-Attention initialized with (input_dim: {self.mha.input_dim}, attn_dim={self.mha.attn_embedding_dim}, {self.mha.num_heads} heads")
        print(multi_head_attn)
        print(multi_head_attn.shape)
        d_model = self.input_dim
        vocab_size = 100

        self.feed_forward = nn.Sequential(
            nn.Linear(self.input_dim, self.ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.ff_hidden_dim, self.input_dim)
        )

        
        normalization_layer1 = nn.LayerNorm(d_model)

        normalization_layer2 = nn.LayerNorm(d_model)

        norm_attn = self.normalization_layer1(
                 + multi_head_attn
            )
        

        ff_norm = self.feed_forward(norm_attn)
        norm_attn_2 = normalization_layer2(ff_norm + norm_attn)
        y = norm_attn_2
        return y





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

out = encoder.forward()


print("\n\n\n")
print(out)
