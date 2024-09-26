import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_function(x_recon, x, mu, sigma):
    # 복원 오차 (MSE)
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
    
    # KLD (Kullback-Leibler Divergence)
    kld_loss = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
    
    return recon_loss + kld_loss

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.01)  # Truncated normal 초기화
            nn.init.constant_(m.bias, 0.01)           # 편향을 0.01로 초기화