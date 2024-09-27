import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import os
import numpy as np
import argparse
import yaml

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.vae import *
from utils import *

class VAE_Trainer():
    def __init__(self):
        self.args = self.parse_args()
        self.device = 'cuda:0'
        # self.device = 'cpu'
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        self.init_data()
        self.init_model()
        self.init_optimizer()
        self.init_logger()

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Train a reinforcement learning agent")
        parser.add_argument("--version", type=str, help="Version of model", default='model')
        parser.add_argument("--predict_mode", type=str, help="predict_mode of model", default='delta')
        parser.add_argument("--env_name", type=str, help="Environment name", default='MountainCar-v0')
        parser.add_argument("--data_name", type=str, help="Expert's data name")
        parser.add_argument("--n_state", type=int, help="n_state", default=2)
        parser.add_argument("--n_z", type=int, help="n_z", default=2)
        parser.add_argument('--units', nargs='*', type=int, help='Input units list of integers')
        parser.add_argument("--batch", type=int, help="batch size", default=32)
        parser.add_argument("--epochs", type=int, help="epochs", default=1000)
        parser.add_argument("--lr", type=float, help="learning rate of optimizer", default=0.0002)
        parser.add_argument("--patience", type=int, help="patience for early stop", default=30)
        parser.add_argument("--val_ratio", type=float, help="Ratio of validation data", default=0.1)
        args = parser.parse_args()
        return args

    def init_data(self):
        data_path = os.path.join(os.getcwd(), 'data', 'state_pairs', self.args.env_name, self.args.data_name)
        data = np.load(data_path)

        val_idx = np.random.choice(len(data), int(len(data)*self.args.val_ratio), replace=False)
        train_idx = np.setdiff1d(np.arange(len(data)), val_idx)

        train_data, val_data = data[train_idx], data[val_idx]

        del data

        train_dataset = Data(train_data)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch, shuffle=True, num_workers=8)
        val_dataset = Data(val_data)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.args.batch, shuffle=True, num_workers=8)
    
    def init_model(self):
        self.model = VAE(
            input_dim=self.args.n_state,
            hidden_dim=self.args.units,
            device=self.device,
            z_dim=1
        )

        self.model.to(device=self.device)

        # for name, param in self.model.named_parameters():
        #     print(f"Parameter '{name}' is on device: {param.device}")
    
    def init_optimizer(self):
        # Initialize optimizers
        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
        self.optimizer = optim.Adam(params, lr=self.args.lr)

    def init_logger(self):
        # Initialize TensorBoard logger
        log_dir = os.path.join(os.getcwd(), 'logs', self.args.env_name, 'vae', self.args.version)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        train_progress = tqdm(self.train_dataloader, desc=f"Epoch [{epoch+1}/{self.args.epochs}]")

        for i, batch in enumerate(train_progress):
            s = batch['state'].to(device=self.device)
            if self.args.predict_mode == 'next':
                s_next = batch['next_state'].to(device=self.device)
            else:
                s_next = batch['delta_s'].to(device=self.device)

            s_recon, mu, sigma = self.model(s)

            self.optimizer.zero_grad()

            loss_recon = reconstruction_loss(s_next, s_recon)
            loss_kl = kl_divergence(mu, sigma)

            loss = loss_recon + loss_kl

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            train_progress.set_postfix(Train_Loss=running_loss/(i+1))

            # Log losses to TensorBoard
            self.writer.add_scalar('Train/Loss_recon', loss_recon.item(), epoch * len(self.train_dataloader) + i)
            self.writer.add_scalar('Train/Loss_kl', loss_kl.item(), epoch * len(self.train_dataloader) + i)

        return running_loss / len(train_progress)

    def validate(self):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in self.val_dataloader:
                s = batch['state'].to(device=self.device)
                if self.args.predict_mode == 'next':
                    s_next = batch['next_state'].to(device=self.device)
                else:
                    s_next = batch['delta_s'].to(device=self.device)
                
                s_recon, mu, sigma = self.model(s)

                loss_recon = reconstruction_loss(s_next, s_recon)
                loss_kl = kl_divergence(mu, sigma)

                loss = (loss_recon + loss_kl) / self.args.batch

                val_loss += loss.item()

        val_loss /= len(self.val_dataloader)
        return val_loss

    def save_model(self):
        self.save_path = os.path.join(os.getcwd(), 'runs', self.args.env_name, 'vae', self.args.version)
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best_model.pth'))
        print('Model saved!')

    def save_config(self):
        config = vars(self.args)
        config['val_loss'] = self.best_val_loss
        with open(os.path.join(self.save_path, 'config.yaml'), 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
    
    def train(self):
        for epoch in range(self.args.epochs):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate()

            # Update progress bar with validation loss
            print(f"Epoch [{epoch+1}/{self.args.epochs}] - Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}")

            # Early stopping and model saving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model()
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.args.patience:
                    self.save_config()
                    print('Early stopping!') 
                    break

        print('Finished Training')

if __name__ == "__main__":
    trainer = VAE_Trainer()
    trainer.train()