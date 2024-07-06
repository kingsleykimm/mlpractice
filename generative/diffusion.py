import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, num_features: int):
        # sinusoidal encoding from transformers
        half_dim = num_features // 2
        emb = torch.log(10000) / (half_dim - 1) # pos term
        emb = torch.exp(torch.arange(half_dim) * -emb)
        
