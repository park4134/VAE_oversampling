import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import tqdm
from glob import glob

import os
import numpy as np
import argparse
import yaml

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models import VAE
from utils import *

class Oversampler():
    def __init__(self):
        self.args = self.parse_args()
        self.device = 'cuda:0'
        self.get_path()
        self.configure()
        self.get_vae()

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Oversampling expert data.")
        parser.add_argument("--env_name", type=str, help="Environment name.", default='MountainCar-v0')
        parser.add_argument("--data_name", type=str, help="Expert's data name")
        parser.add_argument("--n_sample", type=int, help="Number of samples for each states.", default=4)
        parser.add_argument("--model_name", type=str, help="VAE model name.", default='model1')
        args = parser.parse_args()
        return args
    
    def get_path(self):
        self.data_path = os.path.join(os.getcwd(), 'data', 'state_pairs', self.args.env_name, f'{self.args.data_name}.npy')
        self.save_path = os.path.join(os.getcwd(), 'data', 'state_pairs', self.args.env_name)
        self.vae_path = os.path.join(os.getcwd(), 'runs', self.args.env_name, 'vae', self.args.model_name)
        self.filename = f'{self.args.data_name}_oversample_{self.args.n_sample}'
    
    def configure(self):
        with open(os.path.join(self.vae_path, 'config.yaml'), 'r') as file:
            self.config_vae = yaml.safe_load(file)
        self.n_state = self.config_vae['n_state']
        self.units = self.config_vae['units']
        self.n_z = self.config_vae['n_z']

    def get_vae(self):
        self.model_vae = VAE(
            input_dim=self.n_state,
            hidden_dim=self.units,
            device=self.device,
            z_dim=self.n_z
        )
        self.model_vae.load_state_dict(torch.load(os.path.join(self.vae_path, 'best_model.pth'), map_location=self.device))
        self.model_vae.eval().to(device=self.device)
    
    def oversample(self):
        self.expert_data = np.load(self.data_path)
        self.oversample_data = np.array([])

        for s in tqdm(self.expert_data[:,0]):
            s_list = [s]*4 # (n, n_state)
            s_n_list = torch.tensor([])
            s = torch.from_numpy(s).unsqueeze(0)
            for n in range(self.args.n_sample):
                s_recon, mu, sigma = self.model_vae(s.to(device=self.device))
                if n == 0:
                    s_n_list = s_recon # (1, n_state)
                else:
                    s_n_list = torch.cat((s_n_list, s_recon), dim=0)
            # s_n_list : (n, n_state)
            
            s_pair = np.concatenate((np.expand_dims(s_list, axis=1), np.expand_dims(s_n_list.cpu().detach(), axis=1)), axis=1)

            if len(self.oversample_data) == 0:
                self.oversample_data = s_pair
            else:
                self.oversample_data = np.vstack((self.oversample_data, s_pair))
        
        self.save_data()
        print(f"Oversample complete, Length : {len(self.oversample_data)}")
    
    def save_data(self):
        self.oversample_data = np.vstack((self.expert_data, self.oversample_data))
        np.save(os.path.join(self.save_path, self.filename), self.oversample_data)

if __name__=="__main__":
    oversampler = Oversampler()
    oversampler.oversample()