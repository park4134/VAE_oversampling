import torch
import torch.nn as nn
import torch.nn.functional as F

class StateEncoder(nn.Module):
    def __init__(self, n_state: int, units_s_e: list, device: str='cuda:0', alpha: float=0.2):
        super(StateEncoder, self).__init__()
        self.n_state = n_state
        self.units_s_e = units_s_e
        self.device = device
        self.alpha = alpha

        self.hidden_layer = []
        
        for i in range(len(self.units_s_e)):
            if i == 0:
                self.hidden_layer.append(nn.Linear(self.n_state, self.units_s_e[i]))
            else:
                self.hidden_layer.append(nn.Linear(self.units_s_e[i-1], self.units_s_e[i]))
            self.hidden_layer.append(nn.LeakyReLU(self.alpha))
        
        self.hidden_layer = nn.Sequential(*self.hidden_layer)
    
    def forward(self, s):
        s_e = self.hidden_layer(s)

        return s_e
    
class Policy(nn.Module):
    def __init__(self, s_e_dim: int, n_latent_action: int, units_p: list, device: str='cuda:0', alpha: float=0.2):
        super(Policy, self).__init__()
        self.s_e_dim = s_e_dim
        self.units_p = units_p
        self.n_latent_action = n_latent_action
        self.device = device
        self.alpha = alpha
        self.device = device

        self.hidden_layer = []
        
        if len(self.units_p) == 0:
            self.hidden_layer.append(nn.Linear(self.s_e_dim, self.n_latent_action))
        else:
            for i in range(len(self.units_p)):
                if i == 0:
                    self.hidden_layer.append(nn.Linear(self.s_e_dim, self.units_p[i]))
                else:
                    self.hidden_layer.append(nn.Linear(self.units_p[i-1], self.units_p[i]))
                self.hidden_layer.append(nn.LeakyReLU(self.alpha))
            self.hidden_layer.append(nn.Linear(self.units_p[-1], self.n_latent_action))

        # self.hidden_layer.append(nn.Softmax())

        self.hidden_layer = nn.Sequential(*self.hidden_layer)
        
    def forward(self, s_e):
        z_p = self.hidden_layer(s_e)
        return z_p

class Generator(nn.Module):
    def __init__(self, s_e_dim: int, n_state: int, n_latent_action: int, units_g: list, device: str='cuda:0', alpha: float=0.2):
        super(Generator, self).__init__()
        self.s_e_dim = s_e_dim
        self.n_state = n_state
        self.n_latent_action = n_latent_action
        self.units_g = units_g
        self.device = device
        self.alpha = alpha

        self.action_encoder = nn.Sequential(
            nn.Linear(self.n_latent_action, self.s_e_dim),
            nn.LeakyReLU(self.alpha)
        )

        self.concat_layer = nn.Sequential(
            nn.Linear(2*self.s_e_dim, self.s_e_dim),
            nn.LeakyReLU(self.alpha)
        )

        self.hidden_layer = []
        
        for i in range(len(self.units_g)):
            if i == 0:
                self.hidden_layer.append(nn.Linear(self.s_e_dim, self.units_g[i]))
                self.hidden_layer.append(nn.LeakyReLU(self.alpha))
            else:
                self.hidden_layer.append(nn.Linear(self.units_g[i-1], self.units_g[i]))
                self.hidden_layer.append(nn.LeakyReLU(self.alpha))

        self.hidden_layer.append(nn.Linear(self.units_g[-1], self.n_state))
        self.hidden_layer = nn.Sequential(*self.hidden_layer)
        
    def forward(self, s_e):
        delta_s = []

        for z in range(self.n_latent_action):
            # z_onehot = F.one_hot(torch.tensor(z, device=self.device), num_classes=self.n_latent_action).float()
            z_onehot = F.one_hot(torch.tensor(z), num_classes=self.n_latent_action).float()
            z_onehot = z_onehot.unsqueeze(0).expand(s_e.shape[0], -1)

            z_e = self.action_encoder(z_onehot.to(device=self.device))

            concat = torch.cat((s_e, z_e), dim=-1)
            concat = self.concat_layer(concat)

            concat = self.hidden_layer(concat).unsqueeze(0) # (1, batch_size, n_state)
            
            if z == 0:
                delta_s = concat
            else:
                delta_s = torch.cat((delta_s, concat), dim=0)

        return delta_s

class LatentPolicy(nn.Module):
    def __init__(self, n_state: int, n_latent_action: int, units_se: list, units_p: list, units_g: list, device: str='cuda:0', alpha: float=0.2):
        super(LatentPolicy, self).__init__()
        self.n_state = n_state
        self.n_latent_action = n_latent_action
        self.units_se = units_se
        self.units_p = units_p
        self.units_g = units_g
        self.device = device
        self.alpha = alpha
        self.s_e_dim = self.units_se[-1]

        self.state_encoder = StateEncoder(self.n_state, self.units_se, self.device, self.alpha)
        self.policy = Policy(self.s_e_dim, self.n_latent_action, self.units_p, self.device, self.alpha)
        self.generator = Generator(self.s_e_dim, self.n_state, self.n_latent_action, self.units_g, self.device, self.alpha)

        self.state_encoder.to(device=self.device)
        self.policy.to(device=self.device)
        self.generator.to(device=self.device)
    
    def forward(self, x):
        s_e = self.state_encoder(x)
        z_p = self.policy(s_e)
        delta_s = self.generator(s_e)

        return z_p, delta_s
    
if __name__=="__main__":
    n_state = 2
    n_latent_action = 3
    units_se = [32, 64]
    units_p = []
    units_g = [32]
    alpha = 0.2
    model_lp = LatentPolicy(n_state, n_latent_action, units_se, units_p, units_g, 'cuda:0', alpha)

    model_lp.to(device='cuda:0')
    
    states = torch.randn(32, n_state, device='cuda:0')  # Batch size of 32

    from torchsummary import summary
    summary(model_lp, input_size=(2,))

    # Forward pass through the model
    z_p, delta_s = model_lp(states)

    # Check the shapes of the output tensors
    print("z_p shape:", z_p.shape)  # Should be (32, n_latentaction)
    print("delta_s shape:", delta_s.shape)  # Should be (3, 32, n_state)