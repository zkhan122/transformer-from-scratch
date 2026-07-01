import random
import math
import re

import torch
import torch.nn as nn

"""
In Large Language Models (LLMs), Vocabulary Size is the total pool of distinct tokens 
(words or subwords) the model knows, 
while Sequence Length is the number of tokens it processes in a single context
 window. A larger vocabulary shortens text, but increases memory, 
whereas a larger sequence length requires exponentially more compute
"""

class PositionalEmbeddingLayer(nn.Module):
    def __init__(self, d_model, seq_len, vocab_size):
        super().__init__()
        self.seq_len = seq_len # number of tokens processed in parallel (size of the dictionary of embeddings)
        self.d_model = d_model # the size of each embedding vector
        self.vocab_size = vocab_size
        # input_matrix = torch.matmul(self.d_model, self.seq_len) # word2vec 
        self.input_embeddings = nn.Embedding(self.vocab_size, self.d_model, _freeze=False)
        weight_embeddings = self.get_positional_encoding(self.seq_len, self.d_model)
        self.positional_embeddings = nn.Embedding(self.seq_len, 
                                                self.d_model, 
                                                _weight=weight_embeddings,
                                                _freeze=True)
    
    def token_mapping(self, seq): # lookup table (dictionary for token -> id)
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
                positional_enc[position][2*i] = torch.sin(torch.tensor(position/(math.pow(n, (2*i)/d_model))))
                positional_enc[position][(2*i)+1] = torch.cos(torch.tensor(position/(math.pow(n, (2*i+1)/d_model))))
        return positional_enc
    