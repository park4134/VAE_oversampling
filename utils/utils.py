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

def min_action(next_state, next_state_hat):  
    difference = next_state_hat - next_state

    squared_difference = difference ** 2

    sum_of_squares = torch.sum(squared_difference, dim=-1)
    distances = torch.sqrt(sum_of_squares)
    min_actions = torch.argmin(distances, dim=0)
    return min_actions