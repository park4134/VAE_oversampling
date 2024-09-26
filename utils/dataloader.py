import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from tqdm.auto import tqdm
import os
import numpy as np

class Data(Dataset):
    def __init__(self, data):
        self.data = data
        state, next_state = self.data[0]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        state, next_state = self.data[idx]
        data = {
            "state" : state,
            "next_state" : next_state,
            "delta_s": np.array(next_state) - np.array(state),
        }
        return data