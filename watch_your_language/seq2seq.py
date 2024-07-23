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
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers, dropout=0.2)
        self.apply(init_weights)
    
    def forward(self, x):
        # x is the sequence
        # we're going to split it by steps
        # x = torch.transpose(x, 0, 1)
        # size (batch_size, seq_len), need to apply embedding on each seq_len
        embs = self.embedding(x.type(torch.int32)) # turns into the embedding dim now, embs now has shape (seq_len, batch_size, embed_dim), which is the correct input size
        outputs, final_hidden = self.rnn(embs)
        return outputs, final_hidden # final_hidden after last token in sequence
class SeqDecoder(nn.Module):
    # Context variable: This it the final layer hidden state of the final token
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super(SeqDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim + hidden_dim, hidden_dim, num_layers, dropout=0.2) # decoder adds a hidden_dim because of we need to include the encoder's output
        self.fc_1 = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size
    def forward(self, x, encoder_state):
        # x.shape == (num_steps, batch_size, vocab_size)
        # nn.GRU, since we didn't specify batch_first=False, the input shape to GRU needs to be (num_steps, batch_size, hidden_dim / vocab_size / input_sizee)
        enc_output, enc_hiddens = encoder_state
        # x = torch.transpose(x, 0, 1)
        embs = self.embedding(x.type(torch.int32)) # (batch-size, vocab_size/input_size) -> (num_steps, batch_size, embed_dim)
        context = enc_output[-1] # final token, final layer hidden state, will have size hidden_dim, need to make it more. 
        # context.shape (batch_size, hidden_dim). need to shape it / repeat it to num_steps
        context = context.repeat(embs.shape[0], 1, 1) # since the size is already (batch_size, hidden_dim), we're just repeating it by num_steps
        # context.shape(num_steps, batch_size, hidden_dim)
        embs_and_context = torch.cat((embs, context), dim=-1) # concat along last dimension, we need to specify that since the dimensions are very similar
        output, final_hiddens = self.rnn(embs_and_context, enc_hiddens)
        # output.shape: (num_steps, batch_size, hidden_dim)
        output = self.fc_1(output) # i
        ouput = torch.transpose(output, 0, 1)
        return output, final_hiddens

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, trg_vocab, embed_dim, hidden_dim, num_layers, pad_token):
        super(Seq2Seq, self).__init__()
        self.enc = SeqEncoder(src_vocab, embed_dim, hidden_dim, num_layers)
        self.dec = SeqDecoder(trg_vocab, embed_dim, hidden_dim, num_layers)
        self.pad_token = pad_token
    
    def forward(self, src, trg):
        outs, hidden = self.enc(src)
        return self.dec(trg, [outs, hidden])
    
    def put_masks(self, Y): # y is the target, has the pads
        # this function takes in the target/original output, and outputs a tensor of size (batch_siez, num_steps), where it tells the index / mask
        mask = torch.ones(Y.size(dim=0), Y.size(dim=1))
        for batch_ind in range(Y.size(dim=0)):
            for step_ind in range(Y.size(dim=1)):
                if Y[batch_ind][step_ind] == self.pad_token:
                    mask[batch_ind][step_ind] = 0
        return mask
    def predict_steps(self, num_steps, batch, outputs):
        
        src, model_inputs = batch
        # the shapes will be (batch_size, num_steps, vocab_size)
        outs, hidden = self.enc(src)
        # print(src.shape, outs.shape, hidden.shape)
        for ind in range(num_steps):
            # outputs[i] should be of the 
            dec_out, dec_hidden = self.dec(outputs[-1], [outs, hidden])
            outputs.append(dec_out.argmax(2)) # num_steps, batch_size, vocab_size
        # batch_size, num_steps, 
        return torch.cat(outputs[1:]) # all the outputs, now the size will be num_steps, batch_size, vocab_size
            
        
        