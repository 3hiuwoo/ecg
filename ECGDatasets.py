from wfdb import rdrecord
import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

class CINC2017Dataset(Dataset):
    
    def __init__(self, ecg_dir='training2017', ann_dir='REFERENCE-v3.csv',
                 seg=10, sf=300, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.seg = seg
        self.data = self.get_data(ann_dir, ecg_dir, seg, sf)
      
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        ecg = self.data.iloc[idx, 1:-1].values.astype(float)
        label = int(self.data.iloc[idx, -1])
        if self.transform:
            ecg = self.transform(ecg)
        if self.target_transform:
            label = self.target_transform(label)
        return ecg, label
    
    def get_label(self, ann_dir):
        '''
        read the label file and return the label dataframe
        '''
        labels = pd.read_csv(ann_dir, header=None)
        labels.columns = ['head', 'label']
        label_dict = {'N': 0, 'A': 1, 'O': 2, '~': 3}
        labels.replace(label_dict, inplace=True)
        return labels
    
    def get_data(self, ann_dir, ecg_dir, seg, sf):
        npo = seg * sf
        
        labels = self.get_label(ann_dir)
        
        with open(os.path.join(ecg_dir, 'RECORDS'), 'r') as f:
            heads = f.readlines()
        heads = [head.strip() for head in heads]
        
        signals = {}
        for head in heads:
            record = rdrecord(os.path.join(ecg_dir, head))
            signal = record.p_signal
            signal = np.array(signal).reshape(-1)
            signals[head] = signal
        
        data = {}
        for k, v in signals.items():
            cnt = len(v) // npo
            if cnt == 0:
                continue
            data[k] = [v[cnt*3000:(cnt+1)*3000] for cnt in range(cnt)]
            
        df = pd.DataFrame.from_dict(data, orient='index').reset_index()
        df = df.melt(id_vars='index', value_name='signal').sort_values(
            by=['index', 'variable']).drop('variable', axis=1).dropna()
        flatten = df['signal'].apply(pd.Series)
        df = pd.concat([df.drop('signal', axis=1), flatten], axis=1)
        df.reset_index(drop=True, inplace=True)
        df.columns = ['head'] + list(range(npo))
        
        seg_data = df.merge(labels, on='head')
        
        return seg_data