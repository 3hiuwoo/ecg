import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from utils import get_data

class CINC2017Dataset(Dataset):
    '''
    pytorch implementation of the dataset of cinc2017 
    '''
    def __init__(self, ecg_dir='training2017', ann_dir='data/REFERENCE-v3.csv',
                 seg=10, sf=300, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = get_data(ann_dir, ecg_dir, seg, sf)
      
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        ecg = self.data.iloc[idx, 1:-1].values.astype(float)
        ecg = np.expand_dims(ecg, axis=0)
        label = int(self.data.iloc[idx, -1])
        if self.transform:
            ecg = self.transform(ecg)
        if self.target_transform:
            label = self.target_transform(label)
        return ecg, label
    
    