from models import MultiLayerGRU
import torch.nn as nn
import torch
import torch.nn.functional as F
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)

# nn.GRU format: outputs both the final hidden state (output) and the num_layers next hiddens

class SeqEncoder(nn.Module):
    """For this one, we're going to use the GRU are as an encoder. There are multiple design choices to be made here, but usually the encoded hidden
    state will be the final hidden time step of the source input sequence. The encoder's function is to transform the source/input sequence into
    a fixed length context variable.
    Here the input dim == embed_size which is what we should use from now on for nn.Embedding classes. This dim is different from the hidden_dim.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super(SeqEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim) # constructs matrix of vocab_size * hidden_dim, and is basically a simplified version
        # of nn.Linear
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers, dropout=0.1)
        self.apply(init_weights)
    def forward(self, x):
        # x is the sequence
        # we're going to split it by steps
        embs = self.embedding(x) # turns into the embedding dim now, embs now has shape (seq_len, batch_size, embed_dim)
        outputs, final_hidden = self.rnn(embs)
        return outputs, final_hidden # final_hidden after last token in sequence
class SeqDecoder(nn.Module):
    # Context variable: This it the final layer hidden state of the final token
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super(SeqDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim + hidden_dim, hidden_dim, num_layers, dropout=0.1) # decoder adds a hidden_dim because of we need to include the encoder's output
        self.fc_1 = nn.Linear(hidden_dim, vocab_size)
        self.apply(init_weights)
    def init_hidden_state(self, enc_output):
        self.enc_output, self.enc_hiddens = enc_output

    def forward(self, x):
        # x.shape == (num_steps, batch_size, vocab_size)
        embs = self.embedding(x) # (batch-size, vocab_size/input_size) -> (num_steps, batch_size, embed_dim)
        context = self.enc_output[-1] # final token, final layer hidden state, will have size hidden_dim, need to make it more. 
        # context.shape (batch_size, hidden_dim). need to shape it / repeat it to num_steps
        context = context.repeat(embs.shape[0], 1, 1) # since the size is already (batch_size, hidden_dim), we're just repeating it by num_steps
        embs_and_context = torch.cat((embs, context), dim=-1) # concat along last dimension
        output, final_hiddens = self.rnn(embs_and_context, self.enc_hiddens)
        output = F.softmax(self.fc_1(output), dim=-1)
        return output, final_hiddens

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        self.enc = SeqEncoder(vocab_size, embed_dim, hidden_dim, num_layers)
        self.dec = SeqDecoder(vocab_size, embed_dim, hidden_dim, num_layers)
    
    def forward(self, seq):
        self.dec.init_hidden_state(self.enc(seq))
        return self.dec(seq)
    
    def loss(self, y, y_hat, pad_token): # loss function with masking
        # find the CE loss but mask out the padded tokens
        # both y and y_hat will be of shape (seq_len, batch_size, vocab_size), but we want to mask out the <pad> token
        # CE Loss formula sum(targ prob * log prob of y_hat )
        # calculate the CE loss for each timestep, then return a loss tensor of (seq_len)
        l = y * torch.log(y_hat)
        # mask will be a tensor/list of length seq_len telling which indicies are <pad> tokens
        mask = 