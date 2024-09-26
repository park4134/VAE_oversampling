import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: list, z_dim: int=1):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        self.input_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim[0]),
            nn.ReLU()
        )
        
        self.mu = nn.Sequential(
            nn.Linear(self.hidden_dim[-1], self.z_dim),
            nn.ReLU()
        )

        self.sigma = nn.Sequential(
            nn.Linear(self.hidden_dim[-1], self.z_dim),
            nn.ReLU()
        )

        self.hidden_layer = []
        
        for i in range(len(self.hidden_dim) - 1):
            self.hidden_layer.append(nn.Linear(self.hidden_dim[i], self.hidden_dim[i+1]))
            self.hidden_layer.append(nn.ReLU())
        
        self.hidden_layer = nn.Sequential(*self.hidden_layer)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        z = self.reparameterization(mu, sigma)
        return z, mu, sigma

    def reparameterization(self, mu, sigma):
        std = torch.exp(sigma/2)
        eps = torch.randn_like(std)
        return mu + eps * std

class Decoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: list, z_dim: int=1):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        self.input_layer = nn.Sequential(
            nn.Linear(self.z_dim, self.hidden_dim[-1]),
            nn.ReLU()
        )

        self.hidden_layer = []
        
        for i in range(len(self.hidden_dim) - 1, 0, -1):
            self.hidden_layer.append(nn.Linear(self.hidden_dim[i+1], self.hidden_dim[i]))
            self.hidden_layer.append(nn.ReLU())
        
        self.hidden_layer = nn.Sequential(*self.hidden_layer)
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        return x

class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: list, z_dim: int=1):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, z_dim)
        self.decoder = Decoder(input_dim, hidden_dim, z_dim)
    
    def forward(self, x):
        z, mu, sigma = self.encoder(x)
        
        x_recon = self.decoder(z)

        return x_recon, mu, sigma