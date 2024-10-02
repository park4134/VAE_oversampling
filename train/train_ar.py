import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import sys
import csv
import yaml
import random
import argparse
import numpy as np
import gymnasium as gym

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models import LatentPolicy, ActionRemap
from utils import *

from collections import deque
from tqdm.auto import tqdm
from glob import glob

class AR_Trainer():
    def __init__(self):
        self.args = self.parse_args()
        self.device = 'cuda:0'
        # self.device = 'cpu'
        self.best_val_reward = float('inf')
        self.patience_counter = 0
        self.train_step = 0
        self.eps = self.args.eps
        self.que = deque(maxlen=50000)
        self.criterion = nn.CrossEntropyLoss()
        self.env = gym.make(self.args.env_name, render_mode='rgb_array')
        self.valid_env = gym.make(self.args.env_name, render_mode='rgb_array')

        self.configure()
        self.init_model()
        self.init_optimizer()
        self.init_logger()

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Train a reinforcement learning agent")
        parser.add_argument("--env_name", type=str, help="Environment name", default='MountainCar-v0')
        parser.add_argument("--lp_name", type=str, help="Latent policy name", default='model1')
        parser.add_argument("--n_action", type=int, help="n_action", default=2)
        parser.add_argument('--units', nargs='*', type=int, help='Input units list of integers')
        parser.add_argument("--batch", type=int, help="batch size", default=32)
        parser.add_argument("--steps", type=int, help="steps", default=10000)
        parser.add_argument("--lr", type=float, help="learning rate of optimizer", default=0.002)
        parser.add_argument("--patience", type=int, help="patience for early stop", default=30)
        parser.add_argument("--eps", type=float, help="initial epsilon of random action", default=0.95)
        parser.add_argument("--eps_decay", type=float, help="learning rate of optimizer", default=0.999)
        parser.add_argument("--eps_min", type=float, help="initial epsilon of random action", default=0.05)
        args = parser.parse_args()
        return args

    def configure(self):
        self.model_lp_path = os.path.join(os.getcwd(), 'runs', self.args.env_name, 'latent_policy', self.args.lp_name)
        with open(os.path.join(self.model_lp_path, 'config.yaml'), 'r') as file:
            self.config_lp = yaml.safe_load(file)
        self.n_state = self.config_lp['n_state']
        self.n_latent_action = self.config_lp['n_latent_action']
        self.alpha = self.config_lp['alpha']
    
    def init_model(self):
        self.model_lp = LatentPolicy(
            n_state=self.n_state,
            n_latent_action=self.n_latent_action,
            units_se=self.config_lp['units_se'],
            units_p=self.config_lp['units_p'],
            units_g=self.config_lp['units_g'],
            device=self.device,
            alpha=self.alpha
        )
        self.model_lp.load_state_dict(torch.load(os.path.join(self.model_lp_path, 'best_model.pth'), map_location=self.device))
        self.model_lp.eval().to(device=self.device)

        self.model_ar = ActionRemap(
            n_state=self.n_state,
            n_action=self.args.n_action,
            n_latent_action=self.n_latent_action,
            units=self.args.units,
            units_se=self.config_lp['units_se'],
            device=self.device,
            alpha=self.alpha
        )

        self.model_ar.to(device=self.device)
        initialize_weights(self.model_ar)

        # for name, param in self.model.named_parameters():
        #     print(f"Parameter '{name}' is on device: {param.device}")
    
    def init_optimizer(self):
       self.optimizer = optim.Adam(self.model_ar.parameters(), lr=self.args.lr)

    def init_logger(self):
        num = len(glob(os.path.join(os.getcwd(), 'runs', self.args.env_name, 'action_remap', 'model*')))
        self.save_path = os.path.join(os.getcwd(), 'runs', self.args.env_name, 'action_remap', f'model{num+1}')
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

        self.log_reward = os.path.join(self.save_path, f'validation_log.csv')
        with open(self.log_reward, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['step', 'validation_reward'])

        self.log_action = os.path.join(self.save_path, f'validation_action.csv')
        with open(self.log_action, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['step', 'action'])

    def train_one_episode(self):
        state, info = self.env.reset()

        while self.train_step <= self.args.steps:
            self.train_step += 1
            self.model_ar.train()
            if np.random.uniform(0,1) <= self.eps:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    state_input = state
                    input_s = torch.from_numpy(state_input).unsqueeze(0).to(device=self.device)

                    z_p, delta_s_hat = self.model_lp(input_s)

                    action = self.model_ar(input_s, z_p) # softmax
                    action = torch.argmax(action, dim=1).item()
            next_state, reward, terminated, truncated, info = self.env.step(action)

            self.que.append((state, next_state, action))

            if len(self.que) >= self.args.batch:
                self.eps = max(self.args.eps_min, self.eps * self.args.eps_decay)

                batch = random.sample(self.que, self.args.batch)
                states, next_states, actions = zip(*batch)

                states = torch.tensor(np.array(states), device = self.device)
                next_states = torch.tensor(np.array(next_states), device = self.device)            

                with torch.no_grad():
                    z_p, delta_s_hat = self.model_lp(states)

                next_states_hat = states.unsqueeze(0) + delta_s_hat
                min_actions = min_action(next_states, next_states_hat)
                one_hot_min_actions = torch.zeros_like(z_p)
                one_hot_min_actions[torch.arange(z_p.shape[0]), min_actions] = 1.0

                self.optimizer.zero_grad()
                state_action = self.model_ar(states, one_hot_min_actions)
                loss = self.criterion(state_action, torch.tensor(actions).to(device=self.device))
                loss.backward()
                self.optimizer.step()
            
            if terminated or truncated:
                break
            else:
                state = next_state

    def validate(self):
        tot_valid_rewards = []
        actions = []
        for n in range(50):  # Run for 50 iterations
            valid_rewards = 0.0
            valid_state, valid_info = self.valid_env.reset()
            valid_pre_state = valid_state
            while True:
                self.model_ar.eval()

                with torch.no_grad():
                    valid_state_input = torch.from_numpy(valid_state).unsqueeze(0).to(device=self.device)
                    valid_z_p, valid_delta_s_hat = self.model_lp(valid_state_input)

                    valid_action = self.model_ar(valid_state_input, valid_z_p) # softmax

                    valid_action_one_hot = torch.argmax(valid_action, dim=1).item()
                    valid_next_state, valid_reward, valid_terminated, valid_truncated, valid_info = self.valid_env.step(valid_action_one_hot)

                    actions.append(valid_action_one_hot)
                valid_rewards += valid_reward
                
                with open(self.log_action, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([self.train_step, valid_action_one_hot])
                    
                if valid_terminated or valid_truncated:
                    break
                valid_state = valid_next_state
                
            with open(self.log_reward, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([self.train_step, valid_rewards])
            
            tot_valid_rewards.append(valid_rewards)
        for i in range(self.valid_env.action_space.n):
            print(f"{i} : {len(np.where(np.array(actions) == i)[0])}", end = '\t')
        print('\n')
        return tot_valid_rewards

    def save_model(self):
        torch.save(self.model_ar.state_dict(), os.path.join(self.save_path, 'best_model.pth'))
        print('Model saved!')

    def save_config(self, max_reward):
        config = vars(self.args)
        config['n_state'] = self.n_state
        config['n_latent_action'] = self.n_latent_action
        config['best_val_reward'] = float(max_reward)
        config['alpha'] = self.alpha

        with open(os.path.join(self.save_path, 'config.yaml'), 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
    
    def train(self):
        cnt = 0
        total_validation_reward = [-200.0]
        while self.train_step <= self.args.steps:
            self.train_one_episode()
            tot_valid_rewards = self.validate()
            if len(total_validation_reward) > 0 and np.mean(tot_valid_rewards) > max(total_validation_reward):
                self.save_model()
                self.save_config(max(total_validation_reward))
                cnt = 0
            else:
                cnt += 1

            if cnt > self.args.patience:
                self.save_config(max(total_validation_reward))
                break
                
            total_validation_reward.append(np.mean(tot_valid_rewards))
            print(f'Validation Reward:{np.mean(tot_valid_rewards)}')
        print('Finished Training')

if __name__ == "__main__":
    trainer = AR_Trainer()
    trainer.train()