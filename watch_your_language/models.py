import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: DRNN (MultiLayer RNN), BRNN (bi-directional RNN), Design LR annealer

# we need activation layers to create nonlinear neural networks
class RecurrentLayer(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(RecurrentLayer, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.lin1 = nn.Linear(input_size + hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.lin2 = nn.Linear(hidden_dim, output_size)
    def forward(self, inp, hidden_state):
        # inp.shape = (batch_size, vocab_size)
        input_plus_activation = torch.cat((inp, hidden_state), 1)
        next_hidden = self.lin1(input_plus_activation)
        next_hidden = self.tanh(next_hidden) # (batch_size, hidden_state_size)
        output = self.lin2(next_hidden)
        output = F.softmax(output, dim=1)
        return output, next_hidden
    def batch_train(self, inps, targs, batch_size, seq_len):
        hidden_state = torch.zeros(batch_size, self.hidden_dim)
        loss = 0
        for i in range(seq_len):
            output, hidden_state = self.forward(inps[i], hidden_state)
            loss += batch_ce_loss(inps[i], targs[i])
        return loss
class MultiLayerRNN(nn.Module): 
    # might be unnecessary for our use case + a lot of compute
    def __init__(self, input_size, activation_size, output_size, num_layers):
        super(MultiLayerRNN, self).__init__()
        self.input_size = input_size
        self.activation_size = activation_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.rec_layers = nn.ModuleList([
            RecurrentLayer(input_size, activation_size, output_size) for _ in range(num_layers)
        ])
    def forward(self, x):
        # flow goes downward and rightward
        pass
    def loss(self, y, y_tilda):
        # y is targ, y_tilda is model output, and we iterate across all timesteps
        # y and y_tilda should be tensors of length num_timesteps
        return F.mse_loss(y, y_tilda)
    
class LSTM(nn.Module):
    """Notes on LSTM:
    The C/internal_state/memory_state is like the main highway that all information flows through, the different gates will have different purposes to modify it,
    but it is generally the long-term memory of the cell. The hidden states help to bridge short-term memory gaps, maybe like 10 - 50 tokens apart, but the internal 'highway'
    of c states help bridge long-term dependencies
    """
    def __init__(self, input_dim, hidden_dim, internal_dim):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.internal_dim = internal_dim
        self.forget = nn.Linear(input_dim + hidden_dim, internal_dim)
        self.input = nn.Linear(input_dim + hidden_dim, internal_dim)
        self.input_node = nn.Linear(input_dim + hidden_dim, internal_dim)
        self.output_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.last_fc1 = nn.Linear(hidden_dim, input_dim)

    def forward(self, inp, prev_hidden, internal_state):
        input_hidden = torch.cat((inp, prev_hidden), dim=1) # skip batch_size dim
        forget_output = F.sigmoid(self.forget(input_hidden))
        internal_state = internal_state * forget_output
        input_gate_output = F.sigmoid(self.input(input_hidden))
        input_node_output = F.tanh(self.input_node(input_hidden))
        input_combined = input_gate_output * input_node_output
        internal_state = internal_state + input_combined

        output = F.sigmoid(self.output_gate(input_hidden))
        next_hidden = output * F.tanh(internal_state)
        out_word = F.sigmoid(self.last_fc1(next_hidden))
        return out_word, next_hidden, internal_state
    def batch_train(self, inps, targets, batch_size, seq_len):
        # hidden states are reset every 'sequence'/epoch to capture new information and train the forget gate
        hidden_state = torch.zeros(batch_size, self.hidden_dim)
        internal_state = torch.zeros(batch_size, self.internal_dim)
        loss = 0
        for i in range(seq_len):
            output, hidden_state, internal_state = self.forward(inps[i], hidden_state, internal_state)
            loss += batch_ce_loss(output, targets[i])
        return loss


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRU, self).__init__()
        self.input_dim = input_dim

        self.hidden_dim = hidden_dim
        self.activation = {
            'sigmoid' : nn.Sigmoid(),
            'tanh' : nn.Tanh(),
            'relu' : nn.ReLU()
        }
        # update gate to tell us how much of the past we're counting
        self.update_gate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            self.activation['sigmoid']
        )
        # do we drop some of the previous info? using the reset gate?
        self.reset_gate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            self.activation['sigmoid']
        )
        self.candidate_gate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            self.activation['tanh']
        )
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, inp, prev_hidden):
        concat = torch.cat((inp, prev_hidden), dim=1) # across batch_size dim
        reset = self.reset_gate(concat)
        update = self.update_gate(concat)
        after_reset = reset * prev_hidden
        candidate_hidden = torch.cat((inp, after_reset), dim=1)
        candidate_hidden = self.candidate_gate(candidate_hidden)
        new_hidden = update * prev_hidden + (torch.ones(self.hidden_dim)-update) * candidate_hidden
        out = F.sigmoid(self.output(new_hidden))
        return out, new_hidden
    def batch_train(self, inps, targets, batch_size, seq_len):
        loss = 0
        hidden_state = torch.zeros(batch_size, self.hidden_dim)
        for i in range(seq_len):
            hidden_state, out = self.forward(inps[i], hidden_state)
            loss += batch_ce_loss(out, targets[i])
        return loss



def batch_ce_loss(pred : torch.Tensor, targ : torch.Tensor) -> float:
    # both are sizes of (batch_size, vocab_size)
    batch_size = pred.size(dim=0)
    batch_loss = 0
    for i in range(batch_size):
        pred_word, targ_word = pred[i], targ[i]
        batch_loss += -1 * torch.sum(targ_word * torch.log(pred_word))
    return batch_loss / batch_size