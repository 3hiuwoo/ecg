import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from wfdb import rdrecord


class CINC2017Dataset(Dataset):
    '''CINC2017 Dataset
    
    Args:
        root(str/'pathlib.Path', optional): Root directory of dataset fold, 
            which has regular name 'training2017', will search in present
            directory by default.
        ann_dir(str/'pathlib/Path', optional): Root directory of the csv file
            including each signal's label, will search in present directory by
            default.
        seg(int, optional): specify how long is each segmented signal
        sf(int, optional): sample rate of the signals, which is 300Hz in CINC2017.
        transform (callable, optional): A function/transform that takes in a
            signal array and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
    '''
    def __init__(self, root='training2017', ann_dir='data/REFERENCE-v3.csv',
                 seg=10, sf=300, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = self._load_data(root=root, ann_dir=ann_dir, seg=seg, sf=sf)
        self.classes = ['N', 'A', 'O', '~']
      
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        ecg = self.data.iloc[idx, 1:-1].values.astype(float)
        ecg = np.expand_dims(ecg, axis=0)
        label = self.data.iloc[idx, -1]
        if self.transform:
            ecg = self.transform(ecg)
        if self.target_transform:
            label = self.target_transform(label)
        return ecg, label
    
    
    def _load_label(self, ann_dir):
        '''
        read the label file and return the label dataframe
        '''
        labels = pd.read_csv(ann_dir, header=None)
        labels.columns = ['head', 'label']
        label_dict = {'N': 0, 'A': 1, 'O': 2, '~': 3}
        labels.replace(label_dict, inplace=True)
        return labels


    def _load_data(self, root, ann_dir, seg, sf):
        '''
        read and segment all signals with label
        '''
        # total points of each segment
        npo = seg * sf
        labels = self._load_label(ann_dir)
        
        # read all signals' name
        with open(os.path.join(root, 'RECORDS'), 'r') as f:
            heads = f.readlines()
        heads = [head.strip() for head in heads]
        
        # use wfdb to read .mat file
        signals = {}
        for head in heads:
            signal = rdrecord(os.path.join(root, head)).p_signal.reshape(-1)
            signals[head] = signal

        # segment each signal
        data = {}
        for head, signal in signals.items():
            cnt = len(signal) // npo
            if cnt == 0:
                continue
            data[head] = [signal[cnt*3000:(cnt+1)*3000] for cnt in range(cnt)]

        # let each item in the dict be a row having a variable length with NaN
        # for fill, where each element is a array representing the segmented
        # ecg signal
        df = pd.DataFrame.from_dict(data, orient='index').reset_index()
        
        # divide each row into several rows representing single signal where
        # each element is a entry of the signal array.
        df = df.melt(id_vars='index', value_name='signal').sort_values(
            by=['index', 'variable']).drop('variable', axis=1).dropna()
        flatten = df['signal'].apply(pd.Series)
        df = pd.concat([df.drop('signal', axis=1), flatten], axis=1)
        df.reset_index(drop=True, inplace=True)
        df.columns = ['head'] + list(range(npo))
        
        ## annotate each signal
        seg_data = df.merge(labels, on='head')

        return seg_data
    
    
    def write_csv(self, root):
        self.data.to_csv(root)
        



def load_cinc2017(batch_size, ratio=0.8, shuffle=True,
                  root='training2017', ann_dir='data/REFERENCE-v3.csv',
                  seg=10, sf=300, transform=None, target_transform=None):
    '''
    split the dataset into train and valid and return the dataloader
    '''
    dataset = CINC2017Dataset(root=root, ann_dir=ann_dir, seg=seg, sf=sf,
                              transform=transform, target_transform=target_transform)
    
    train_size = int(ratio * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(
        dataset,[train_size, valid_size],
        generator = torch.Generator().manual_seed(0))
    
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_iter = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return train_iter, valid_iter
    