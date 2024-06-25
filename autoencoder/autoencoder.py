import torch
import torch.nn as nn
import torch.nn.functional as F
class EncoderBlock(nn.Module):
    def __init__(self, num_features: int, input_dim : int, hidden_dim : int):
        # goes through one hidden layer and then into the feature latent space
        super(EncoderBlock, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.encoder(x)
class DecoderBlock(nn.Module):
    def __init__(self, feature_dim : int, output_dim : int, hidden_dim : int):
        super(DecoderBlock, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.decoder(x)

class Autoencoder(nn.Module):
    def __init__(self, feature_dim : int, hidden_layer_dim : int, input_dim : int):
        super(Autoencoder, self).__init__()
        self.encoder = EncoderBlock(feature_dim, input_dim, hidden_layer_dim)
        self.decoder = DecoderBlock(feature_dim, input_dim, hidden_layer_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
def apply_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)
