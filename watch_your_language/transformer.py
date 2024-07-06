import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# each word in the dictionary / unique words has it's own embedding vector which is of size dict_size
class InputEmbeddings(nn.Module): # first layer to convert input tokens into 
    def __init__(self, d_model : int, dict_size : int): # d_model should usually be 512
        super().__init__(InputEmbeddings, self)
        self.dict_size = dict_size
        self.d_model = d_model
        self.embed = nn.Embedding(
            num_embeddings=dict_size,
            embedding_dim=512,
        )

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len : int, d_model : int, p_dropout : float):
        super().__init__()
        # positional sinusodal embeddings
        self.dropout_layer = nn.Dropout(p_dropout)
        self.seq_len = seq_len
        self.d_model = d_model
        self.pe = torch.zeros(seq_len, d_model) # so each word in this sequence will have it's own d_model size embedding vector
        positions = torch.arange(start=0, end=seq_len, dtype=torch.float32)
        div_term = torch.arange(start=0, end=seq_len, step=2).float() * 1/10000 * 1/d_model   # for each i
        self.pe[:, 0::2] = torch.sin(positions * div_term) # sin's fill up even
        self.pe[:, 1::2] = torch.cos(positions * div_term)

        self.pe = self.pe.unsqueeze(0) # adds dimension of 1 at the front for batch sizes
        # sin -> evens, cos -> odds
    def forward(self, input_embedding):
        # now these input embeddings will have dimensions of seq_len * d_model embedding length, so we want to add along the d_model embedding length dimension
        # specifically, for the position the word in input_embedding is in, we want to add the corresponding position_embedding in
        sum_ = input_embedding + (self.pe[:, :input_embedding.shape[1], :]).requires_grad(False) # first colon signifies all the batches, second colons siginifies the max position that the input carries, last_colon signifies the entire dimension
        return self.dropout_layer(sum_)
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, num_heads : int, d_model : int):

        super().__init__()
        # Scaled Dot-Product Attention
        self.head_dim = d_model / num_heads
        self.key_linear = nn.Linear(d_model, d_model)
        self.query_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)   
        self.w_o = nn.Linear(d_model, d_model) # final one     
        self.d_model = d_model
        self.num_heads = num_heads
    def attention(self, query, key : torch.Tensor, value, mask):

        # transpose for dot products, the dims of key will be: batch_size * n_heads * seq_len * d_k
        # dims of query will be: batch_size * n_heads * seq_len * d_k
        # we want to transpose it to do a dot product with query, so flip 2 and 3
        keys_transposed = key.transpose(-2, -1)
        dot_product = query @ keys_transposed * math.sqrt(1/self.d_model)
        if mask:
            dot_product = dot_product @ mask
        softmax_prod = F.softmax(dot_product)
        return softmax_prod @ value
    def forward(self, q, k, v, mask):
        # input size: batch * seq_len * d_model
        q = self.query_linear(q)
        k = self.key_linear(k)
        v = self.value_linear(v)
        q, k, v = self.split(q), self.split(k), self.split(v)

        output = self.attention(q, k, v, mask)
        output = self.w_o(output)
        return output
        # split into k number of heads
    
    def split(self, tensor):
        batch_size, seq_len, d_model = tensor.size()
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.d_model // self.num_heads) # splits up into the num_heads
        tensor.transpose(1, 2) # moves self.num_heads to the front and keeps seq_len in the back for when we process the nodes
        # new_shape: batch_size * self.num_heads * seq_len * self.d_model // self.num_heads
        return tensor

    def concat(self, tensor):
        batch_size, num_heads, seq_len, d_k = tensor.size()
        tensor = tensor.transpose(1, 2) # switch those back 
        tensor = tensor.contiguous() # need to do this to switch it back because of the transpose, ruins how it is laid out in memory according to stride
        tensor = tensor.view(batch_size, seq_len, d_k * num_heads)
        return tensor


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model : int, hidden_dim : int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LayerNorm(nn.Module):
    # LayerNorm is defined in here as well
    def __init__(self, d_model : int):
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x : torch.Tensor):
        mean = x.mean(dim=-1) # along the last dimension, which makes sense
        std = x.std(dim=-1)
        return ((x - mean) @ self.gamma)  / std + self.beta

    
class EncoderBlock(nn.Module):
    def __init__(self, num_heads : int, d_model : int, p_dropout : float):
        super().__init__(EncoderBlock, self)
        self.attention = MultiHeadAttentionBlock(num_heads, d_model)
        self.feed_forward = FeedForwardBlock(d_model, hidden_dim=2048)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p_dropout)


    def forward(self, x, mask):
        att = self.attention(x, x, x, mask)
        # dropout after every sublayert)
        # residual connection
        x = self.norm1(x + self.dropout1(att))
        ffd = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ffd))
        return x
        

class DecoderBlock(nn.Module):
    def __init__(self, num_heads : int, d_model : int, p_dropout : float):
        super().__init__(DecoderBlock, self)

        self.masked_attention = MultiHeadAttentionBlock(num_heads, d_model)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_dropout)

        self.encoder_attention = MultiHeadAttentionBlock(num_heads, d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p_dropout)

        self.feed_forward = FeedForwardBlock(d_model, hidden_dim=2048)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p_dropout)

    def forward(self, x, src_mask, e_x, targ_mask):
        # create self_attention mask
        masked_output = self.masked_attention(x, x, x, targ_mask)
        x = self.norm1(x + self.dropout1(masked_output))
        enc_att = self.encoder_attention(q=x, k=e_x, v=e_x, mask=src_mask) # sources mask to carry on from encoder
        x = self.norm2(x + self.dropout2(enc_att))
        ffd = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ffd))
        return x
    
class EncoderLayers(nn.Module):
    def __init__(self, num_heads : int, d_model : int, p_dropout : float, n_layers : int, vocab_size : int, seq_len : int):
        super().__init__(self)
        self.layers = nn.ModuleList([
            EncoderBlock(num_heads=num_heads, d_model=d_model, p_dropout=p_dropout) for _ in range(n_layers)
        ])
        self.input_embed = InputEmbeddings(d_model=d_model, dict_size=vocab_size)
        self.pos_embed = PositionalEncoding(seq_len=seq_len, d_model=d_model, p_dropout=p_dropout)

    def forward(self, x):
        x = self.input_embed(x)
        x = self.pos_embed(x)
        for module in self.layers:
            x = module(x)
        return x

class DecoderLayers(nn.Module):
    def __init__(self, num_heads : int, d_model : int, p_dropout : float, n_layers : int, seq_len : int, vocab_size : int):
        super().__init__(self)
        self.layers = nn.ModuleList([
            DecoderBlock(num_heads=num_heads, d_model=d_model, p_dropout=p_dropout) for _ in range(n_layers)
        ])
        self.output_embds = InputEmbeddings(d_model=d_model, dict_size=vocab_size)
        self.positional_encoding = PositionalEncoding(seq_len=seq_len, d_model=d_model, p_dropout=p_dropout)
        self.last_linear = nn.Linear(d_model, vocab_size)


    def forward(self, src_mask, targ, targ_mask, encoder):
        # masks should be position based and mask off any future indices
        targ = self.output_embds(targ)
        targ = self.positional_encoding(targ)
        for layer in self.layers:
            targ = layer(targ, targ_mask, encoder)
        targ = self.last_linear(targ)
        return F.softmax(targ)

class Transformer(nn.Module):
    def __init__(self, num_heads : int, d_model : int, p_dropout : float, n_layers : int, seq_len : int, vocab_size : int):
        super().__init(self)
        self.encoder = EncoderLayers(num_heads=num_heads, d_model=d_model, p_dropout=p_dropout, n_layers=n_layers, vocab_size=vocab_size, seq_len=seq_len)
        self.decoder = DecoderLayers(num_heads=num_heads, d_model=d_model, p_dropout=p_dropout, n_layers=n_layers, seq_len=seq_len, vocab_size=vocab_size)

    def forward(self, src, targ, targ_mask, src_mask):
        # src_mask will be for enc_dec
        # targ_mask will be for self-attention, zero out future indices
        encoder_output = self.encoder(src)
        decoder_output = self.decoder(src_mask, targ, targ_mask, encoder_output)
        return decoder_output
    

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
