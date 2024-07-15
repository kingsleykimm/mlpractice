import torch
import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)

def layer_norm(vector, epsilon):
    mean = torch.mean(vector) # across all dimensions, we're just taking it over batch_size = 1
    var = torch.mean((vector - mean) ** 2).mean()
    return (vector - mean) / torch.sqrt(var + epsilon)

class BatchNorm(nn.Module):
    """Batch Norm between training and predicting is different.
    During training, it only uses the statistics of the minibatches, while during """
    def __init__(self, num_features : tuple, num_dim : int): # input_size will be a tuple
        super(BatchNorm, self).__init__()
        input_size = None
        if num_dim == 2:
            input_size = (1, num_features)
        elif num_dim == 4:
            input_size = (1, num_features, 1, 1) # for broadcasting just make it fit to the number of channels
        self.gamma = nn.Parameter(torch.zeros(*input_size))
        self.beta = nn.Parameter(torch.ones(*input_size))
        self.moving_avg, self.moving_var = torch.zeros(*input_size), torch.ones(*input_size)
        self.epsilon = 1e5
        self.momentum = 0.1
    def forward(self, x, predicting):
        return self.batch_norm(self.epsilon, x, self.gamma, self.beta, predicting, self.momentum)

    def batch_norm(self, epsilon, batch, gamma, beta, predicting, momentum):
        # gamma and beta are the same size as element in batch
        # so if we
        if predicting:
            # if we're predicting, we're doing it over the entire dataset
            return gamma * (batch - self.moving_avg) / torch.sqrt(self.moving_var + epsilon) + beta
        # if we're not in predicting (i.e training), we're going to want to just find the means across the batch_size dim
        else:
            mean, var = None, None
            if len(batch.size) == 2: # i.e fc, (batch_size, fc_layer)
                mean = torch.mean(batch, dim=0)
                var = ((batch - mean) ** 2).mean(dim=0) # variance per batch row
                # apply batch norm equation
            elif len(batch.size) == 4: # i.e conv (batch_size, num_channels, height, width)
                mean = torch.mean(batch, dim=(0, 2, 3), keepdim=True) # now mean will be of size (1, channels, 1, 1)
                var = torch.mean((batch - mean) ** 2, dim=(0, 2, 3), keepdim=True) # broadcasting allows it to just subtract
            self.moving_avg = self.moving_avg * (1 - momentum) + mean * momentum
            self.moving_var = self.moving_var * (1 - momentum) + var * momentum
            batch_normalized = gamma * (batch - mean) / torch.sqrt(var + epsilon) + beta
            return batch_normalized