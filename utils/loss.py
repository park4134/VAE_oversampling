import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_min(delta_s, delta_s_hat):
    delta_s = delta_s.unsqueeze(0)
    all_loss = []
    # 각 출력에 대해 mean squared error를 계산
    for i in range(delta_s_hat.shape[0]):
        loss = F.mse_loss(delta_s_hat[i].unsqueeze(0), delta_s, reduction='none')
        all_loss.append(torch.sum(loss, dim=-1))
    # 각 출력에 대한 손실을 쌓고, 최소 손실을 계산
    stacked_min_loss = torch.concat(all_loss, dim=0)
    gen_loss_min = torch.mean(torch.min(stacked_min_loss, dim=0).values)

    return gen_loss_min


def loss_exp(delta_s, z_p, delta_s_hat):
    z_p = F.softmax(z_p, dim=-1)
    expect_delta_s = delta_s_hat * z_p.transpose(0, 1).unsqueeze(-1)

    loss_exp = torch.sum((delta_s - expect_delta_s)**2, dim=1)
    loss_exp = torch.mean(loss_exp)
    return loss_exp

def min_action(next_state, next_state_hat):  
    difference = next_state_hat - next_state

    squared_difference = difference ** 2

    sum_of_squares = torch.sum(squared_difference, dim=-1)
    distances = torch.sqrt(sum_of_squares)
    min_actions = torch.argmin(distances, dim=0)
    return min_actions