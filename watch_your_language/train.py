from torch.utils.data import DataLoader
import torch
from models import *
from datasets import PennTreebank

def train_model(num_epochs=200, lr=1e-3, seq_len=7):
    train_dataset = PennTreebank('../data/', 'ptbdataset/ptb.train.txt', seq_len)
    test_dataset = PennTreebank('../data/', 'ptbdataset/ptb.test.txt', seq_len)

    model = RecurrentLayer(seq_len, seq_len, seq_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# gradient_clipping pattern:
"""
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
optimizer.step()
"""