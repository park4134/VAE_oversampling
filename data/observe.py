from stable_baselines3 import PPO, DQN
from tqdm import tqdm

import gymnasium as gym
import numpy as np
import argparse
import os

class Observer():
    def __init__(self):
        self.args = self.parse_args()
        self.get_path()

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Train a reinforcement learning agent.")
        parser.add_argument("--env_name", type=str, help="Environment name.", default='MountainCar-v0')
        parser.add_argument("--data_name", type=str, help="Expert's data name")
        parser.add_argument("--expert", type=str, help="Expert name.", default='PPO_-200.00')
        parser.add_argument("--method", type=str, help="Expert's method.", default='PPO')
        parser.add_argument("--eps", type=float, help="Noise rate of expert's actions.", default=0.0)
        args = parser.parse_args()
        return args
    
    def get_path(self):
        self.expert_path = os.path.join(os.getcwd(), 'runs', self.args.env_name, 'experts', self.args.expert.split('_')[0], self.args.expert)
        self.save_path = os.path.join(os.getcwd(), 'data', 'state_pairs', self.args.env_name)
        self.filename = f'{self.args.expert}_{self.args.data_len}_{self.args.eps}'
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
    
    def observe(self):
        self.env = gym.make(self.args.env_name, render_mode='rgb_array')

        if self.args.method == 'PPO':
            self.expert = PPO.load(self.expert_path)
        elif self.args.method == 'DQN':
            self.expert = DQN.load(self.expert_path)
        
        state_pairs = []
        rewards = 0.0
        total_rewards =  []

        state, info = self.env.reset()
        pbar = tqdm(range(self.args.data_len))
        for _ in pbar:
            if np.random.uniform(0,1) <= self.args.eps:
                action = self.env.action_space.sample()
            else:
                action, _ = self.expert.predict(state, deterministic=True)
            
            next_state, reward, terminated, truncated, info = self.env.step(action)
            rewards += reward
            state_pairs.append((state, next_state))

            if terminated or truncated:
                state, info = self.env.reset()
                pbar.set_postfix(Reward=rewards)
                rewards = 0.0
            else:
                state = next_state
        
        state_pairs = np.array(state_pairs)
        np.save(os.path.join(self.save_path, self.filename), state_pairs)
        print(f"Saved state pairs to {self.save_path}")

if __name__=='__main__':
    obs = Observer()
    obs.observe()