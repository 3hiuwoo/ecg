import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from utils import get_data
from utils import get_one_hot

class CINC2017Dataset(Dataset):
    '''
    pytorch implementation of the dataset of cinc2017 
    '''
    def __init__(self, ecg_dir='training2017', ann_dir='data/REFERENCE-v3.csv',
                 seg=10, sf=300, transform=None,
                 target_transform=None, train=True):
        self.transform = transform
        self.target_transform = target_transform
        self.data = get_data(ann_dir, ecg_dir, seg, sf)
        self.train = train
      
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        ecg = self.data.iloc[idx, 1:-1].values.astype(float)
        ecg = np.expand_dims(ecg, axis=0)
        ecg = torch.tensor(ecg, dtype=torch.float32)
        label = int(self.data.iloc[idx, -1])
        if self.transform:
            ecg = self.transform(ecg)
        if self.target_transform:
            label = self.target_transform(label)
        return ecg, label


def load_cinc2017(batch_size, ratio=0.8, shuffle=True):
    '''
    split the dataset into train and valid and return the dataloader
    '''
    dataset = CINC2017Dataset(target_transform=get_one_hot)
    
    train_size = int(ratio * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(
        dataset,[train_size, valid_size],
        generator = torch.Generator().manual_seed(0))
    
    train_iter = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=shuffle)
    valid_iter = DataLoader(valid_dataset, batch_size=batch_size,
                            shuffle=shuffle)
    
    return train_iter, valid_iter
    