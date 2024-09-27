import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: list, z_dim: int=1, device: str='cuda:0'):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.device = device

        self.input_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim[0]),
            nn.ReLU(),
            nn.Dropout(p=0.2)
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
            self.hidden_layer.append(nn.Dropout(p=0.2))
        
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
    def __init__(self, input_dim: int, hidden_dim: list, output_dim: int, device: str='cuda:0'):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device

        self.input_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim[-1]),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        self.hidden_layer = []
        
        for i in range(len(self.hidden_dim) - 1, 0, -1):
            self.hidden_layer.append(nn.Linear(self.hidden_dim[i], self.hidden_dim[i-1]))
            self.hidden_layer.append(nn.ReLU())
            self.hidden_layer.append(nn.Dropout(p=0.2))
        self.hidden_layer.append(nn.Linear(self.hidden_dim[0], self.output_dim))
        self.hidden_layer = nn.Sequential(*self.hidden_layer)
        
    def forward(self, x):
        x = self.input_layer(x)
        x = F.sigmoid(self.hidden_layer(x))
        # x = self.hidden_layer(x)
        return x

class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: list, z_dim: int=1, device: str='cuda:0'):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.z_dim = z_dim

        self.encoder = Encoder(self.input_dim, self.hidden_dim, self.z_dim, self.device)
        self.decoder = Decoder(self.z_dim, self.hidden_dim, self.input_dim, self.device)

        self.encoder.to(device=self.device)
        self.decoder.to(device=self.device)
    
    def forward(self, x):
        z, mu, sigma = self.encoder(x)

        x_recon = self.decoder(z)

        return x_recon, mu, sigma

if __name__=="__main__":
    input_dim = 2
    hidden_dim = [64, 32]
    z_dim = 1
    model_lp = VAE(input_dim, hidden_dim, 'cuda:0', z_dim)
    model_lp.to(device='cuda:0')
    
    input_vector = torch.randn(32, input_dim)  # Batch size of 32

    # Forward pass through the model
    x_recon, mu, sigma = model_lp(input_vector.to(device='cuda:0'))

    # Check the shapes of the output tensors
    print("x_recon shape:", x_recon.shape)  # Should be (32, n_latentaction)
    print("mu shape:", mu.shape)  # Should be (3, 32, n_state)
    print("sigma shape:", sigma.shape)  # Should be (3, 32, n_state)