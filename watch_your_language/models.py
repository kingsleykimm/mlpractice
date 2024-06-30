import torch
import torch.nn as nn
import torch.nn.functional as F

# we need activation layers to create nonlinear neural networks
class RecurrentLayer(nn.Module):
    def __init__(self, input_size, activation_size, output_size):
        super(RecurrentLayer, self).__init__()
        self.input_size = input_size
        self.activation_size = activation_size
        self.output_size = output_size
        self.lin1 = nn.Linear(input_size + activation_size, activation_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.lin2 = nn.Linear(activation_size, output_size)
    def forward(self, inp, hidden_state):
        # inp.shape = (batch_size, vocab_size)
        input_plus_activation = torch.cat(inp, hidden_state, dim=1)
        next_hidden = self.lin1(input_plus_activation)
        next_hidden = self.tanh(next_hidden) # (batch_size, hidden_state_size)
        output = self.lin2(next_hidden)
        output = self.tanh(output)
        return output, next_hidden



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
    

# to implement : DRNN, BRNN
class GRU(nn.Module):
    def __init__(self, input_dim, output_dim, c_dim, hidden_dim, num_steps, activation='tanh'):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.c_dim = c_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.activation = {
            'sigmoid' : nn.Sigmoid(),
            'tanh' : nn.Tanh(),
            'relu' : nn.ReLU()
        }
        # update gate to tell us how much of the past we're counting
        self.update_gate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            self.activation[activation]
        )
        # do we drop some of the previous info? using the reset gate?
        self.reset_gate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            self.activation[activation]
        )
        self.candidate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, inp, prev_hidden):
        concat = torch.concat(inp, prev_hidden, dim=1) # across batch_size dim
        reset = self.reset_gate(concat)
        update = self.update_gate(concat)
        after_reset = reset @ prev_hidden
        candidate_hidden = torch.concat(inp, after_reset, dim=1)
        candidate_hidden = self.activation(candidate_hidden)