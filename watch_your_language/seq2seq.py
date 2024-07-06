from models import GRU
import torch.nn as nn
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)


class SeqEncoder(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim):
        super(SeqEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim) # constructs matrix of vocab_size * hidden_dim, and is basically a simplified version
        # of nn.Linear
        self.rnn = GRU(input_dim, hidden_dim)
        self.apply(init_weights)
    def forward(self, x):
        embs = self.embedding(x) # g