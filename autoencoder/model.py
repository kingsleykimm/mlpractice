import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
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
    def __init__(self, output_dim : int, hidden_dim : int, feature_dim : int, n_layers : int):
        super(DecoderBlock, self).__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) if i % 2 == 0 else nn.ReLU() for i in range(n_layers * 2)
        ])
        self.final = nn.Linear(hidden_dim, output_dim)
        self.relu2 = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final(x)
        return self.relu2(x)

class Autoencoder(nn.Module):
    def __init__(self, feature_dim : int, hidden_layer_dim : int, input_dim : int):
        super(Autoencoder, self).__init__()
        self.encoder = EncoderBlock(feature_dim, input_dim, hidden_layer_dim)
        self.decoder = DecoderBlock(feature_dim, input_dim, hidden_layer_dim, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    def calc_loss(self, inputs):
        output = self.forward(inputs)
        return F.mse_loss(output, inputs)

class VariationalEncoder(nn.Module):
    def __init__(self, input_dim : int, hidden_dim : int, latent_dim : int, n_layers : int):
        super(VariationalEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) if i % 2 == 0 else nn.ReLU() for i in range(n_layers * 2)
        ])
        self.latent_dim = latent_dim
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.vars = nn.Linear(hidden_dim, latent_dim)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        for layer in self.hidden_layers:
            x = layer(x)
        means = self.mean(x)
        variances = self.vars(x)
        return means, variances
        
class VAE(nn.Module):
    def __init__(self, input_dim : int, hidden_dim : int, latent_dim : int, n_layers : int):
        super(VAE, self).__init__()
        self.decoder = DecoderBlock(input_dim, hidden_dim, latent_dim, n_layers)
        self.encoder = VariationalEncoder(input_dim, hidden_dim, latent_dim, n_layers)
        self.normal = Normal(torch.zeros(latent_dim), torch.ones(latent_dim))
        self.apply(apply_weights)
    def forward(self, x):
        means, variances = self.encoder(x)
        epsilon = torch.rand(1).item() * 2 - 1
        latents = means + epsilon * torch.exp(variances)
        x = self.decoder(latents)
        return x, means, variances
    def custom_loss(self, target, output, means, variances):
        loss = F.mse_loss(output, target)
        # variances are normal right now, need to be logged down
        kl_loss = -0.5 * torch.sum(-torch.exp(variances) -  means ** 2 + 1 + variances)
        return loss + kl_loss
def apply_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)
