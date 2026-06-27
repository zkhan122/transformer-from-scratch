import random
import math
import re

import torch

class EmbeddingLayer():
    def __init__(self, d_model, seq_len):
        self.d_model = d_model
        self.seq_len = seq_len # number of tokens processed in parallel
        input_matrix = torch.matmul(self.d_model, self.seq_len) # word2vec
    
    def token_mapping(seq): # lookup table (dictionary for token -> id)
        seq_buffer = re.findall(r"[\w']+", seq)
        token_id_buffer = {}
        for i in range(len(seq_buffer)): 
            token = seq_buffer[i]
            if token not in token_id_buffer:
                token_id_buffer[token] = random.randint(1, 10000000000)
        return token_id_buffer
    
    def get_positional_encoding(self, seq_len, d_model, n=10000):
        positional_enc = torch.zeros(seq_len, d_model)
        for position in range(seq_len):
            for i in range(d_model // 2):
                positional_enc[position, 2*i] = torch.sin(torch.tensor(position/(math.pow(n, 2*i/d_model))))
                positional_enc[position, 2*i+1] = torch.cos(torch.tensor(position/(math.pow(n, 2*i+1/d_model))))
        return positional_enc
    