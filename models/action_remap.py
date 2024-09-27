import torch
import torch.nn as nn
import torch.nn.functional as F

class StateEncoder(nn.Module):
    def __init__(self, n_state: int, units: list, device: str='cuda:0', alpha: float=0.2):
        super(StateEncoder, self).__init__()
        self.n_state = n_state
        self.units = units
        self.device = device
        self.alpha = alpha

        self.hidden_layer = []
        
        for i in range(len(self.units)):
            if i == 0:
                self.hidden_layer.append(nn.Linear(self.n_state, self.units[i]))
            else:
                self.hidden_layer.append(nn.Linear(self.units[i-1], self.units[i]))
            self.hidden_layer.append(nn.LeakyReLU(self.alpha))
        
        self.hidden_layer = nn.Sequential(*self.hidden_layer)
    
    def forward(self, s):
        s_e = self.hidden_layer(s)

        return s_e
    
class ActionRemap(nn.Module):
    def __init__(self, n_state: int, n_action: int, n_latent_action: int, units: list, units_se: list, device: str='cuda:0', alpha: float=0.2):
        super(ActionRemap, self).__init__()
        self.n_state = n_state
        self.n_action = n_action
        self.n_latent_action = n_latent_action
        self.units = units
        self.units_se = units_se
        self.device = device
        self.alpha = alpha

        self.state_encoder = StateEncoder(self.n_state, self.units_se, self.device, self.alpha)

        self.action_encoder = nn.Sequential(
            nn.Linear(self.n_latent_action, self.units_se[-1]),
            nn.LeakyReLU(alpha)
        )

        self.hidden_layer = []
        
        for i in range(len(self.units)):
            if i == 0:
                self.hidden_layer.append(nn.Linear(2*self.units_se[-1], self.units[i]))
                self.hidden_layer.append(nn.LeakyReLU(self.alpha))
            else:
                self.hidden_layer.append(nn.Linear(self.units[i-1], self.units[i]))
                self.hidden_layer.append(nn.LeakyReLU(self.alpha))

        self.hidden_layer.append(nn.Linear(self.units[-1], self.n_action[i+1]))
        
        self.hidden_layer = nn.Sequential(*self.hidden_layer)
    
    def forward(self, s, z):
        s_e = self.state_encoder(s)
        z_e = self.action_encoder(z)

        concat = torch.cat((s_e, z_e), dim=-1)
        concat = self.concat_layer(concat)
        
        a = self.hidden_layer(concat)

        return a