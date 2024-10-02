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
        self.state_encoder.to(device=self.device)

        self.action_encoder = nn.Sequential(
            nn.Linear(self.n_latent_action, self.units_se[-1]),
            nn.LeakyReLU(self.alpha)
        )

        self.hidden_layer = []
        
        for i in range(len(self.units)):
            if i == 0:
                self.hidden_layer.append(nn.Linear(2*self.units_se[-1], self.units[i]))
                self.hidden_layer.append(nn.LeakyReLU(self.alpha))
            else:
                self.hidden_layer.append(nn.Linear(self.units[i-1], self.units[i]))
                self.hidden_layer.append(nn.LeakyReLU(self.alpha))

        self.hidden_layer.append(nn.Linear(self.units[-1], self.n_action))
        
        self.hidden_layer = nn.Sequential(*self.hidden_layer)
    
    def forward(self, s, z):
        s_e = self.state_encoder(s)

        z = torch.argmax(z, dim=-1)
        latent_action_onehot = torch.zeros(z.shape[0], self.n_latent_action, device=self.device)
        latent_action_onehot[torch.arange(z.shape[0]), z] = 1

        z_e = self.action_encoder(latent_action_onehot)

        concat = torch.cat((s_e, z_e), dim=-1)
        
        a_p = self.hidden_layer(concat)

        return a_p
    
if __name__=="__main__":
    model_ar = ActionRemap(
        n_state=2,
        n_action=3,
        n_latent_action=3,
        units=[64, 32],
        units_se=[32, 64],
        device='cuda:0',
        alpha=0.2
    )

    model_ar.to(device='cuda:0')

    from torchsummary import summary
    summary(model_ar, input_size=[(2, ), (3, )])
    # states = torch.randn(32, 2)  # Batch size of 32
    # z = torch.randn(32, 3)

    # Forward pass through the model
    # a_p = model_ar(states, z)

    # Check the shapes of the output tensors
    # print("a_p shape:", a_p.shape)  # Should be (32, n_latentaction)