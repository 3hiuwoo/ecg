import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from wfdb import rdrecord


class CINC2017Dataset(Dataset):
    '''CINC2017 Dataset
    
    Args:
        root(path, optional): Root directory of dataset fold, 
            which has regular name 'training2017', will search in present
            directory by default.
        seg(int, optional): specify how long is each segmented signal.
        stride(int, optional): specify the stride of the sliding window.
        sf(int, optional): sample rate of the signals, which is 300Hz in CINC2017.
        transform (callable, optional): A function/transform that takes in a
            signal array and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
    '''
    def __init__(self, root='training2017', seg=10, stride=5, sf=300,
                 transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.label = self._load_label(root)
        self.segments = self._load_data(root=root, seg=seg, stride=stride, sf=sf)
        self.data = pd.merge(self.segments, self.label, on='head')
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
    
    
    def _load_label(self, root):
        '''
        read the label file and return the label dataframe
        '''
        labels = pd.read_csv(os.path.join(root, 'REFERENCE-v3.csv'), header=None)
        labels.columns = ['head', 'label']
        label_dict = {'N': 0, 'A': 1, 'O': 2, '~': 3}
        labels = labels.replace(label_dict)
        return labels


    def _load_data(self, root, seg, stride, sf):
        '''
        read and segment all signals with label
        '''
        # total points of each segment
        segp = seg * sf
        stridep = stride * sf
        
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
            if len(signal) < segp:
                continue
            seg_sigs = [signal[i:i+segp]
                        for i in range(0, len(signal)-segp+1, stridep)]
            data[head] = seg_sigs
                

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
        df.columns = ['head'] + list(range(segp))

        return df
    
    
    def write_csv(self, root):
        self.data.to_csv(root)
        

def load_cinc2017(batch_size, ratio=0.9, shuffle=True, root='training2017',
                  seg=10, stride=5, sf=300,
                  transform=None, target_transform=None):
    '''
    split the dataset into train and valid and return the dataloader
    '''
    dataset = CINC2017Dataset(root=root, seg=seg, stride=stride, sf=sf,
                              transform=transform,
                              target_transform=target_transform)
    if ratio == 1:
        train_iter = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return train_iter, None
    
    train_size = int(ratio * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(
        dataset,[train_size, valid_size])
    
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_iter = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return train_iter, valid_iter
    
    
def get_cinc2017_class(label, one_hot=False):
    '''
    decode the one-hot encoding to class
    
    Args:
        label(array): a single label
        one_hot(bool, optional): specify whether the labels are one-hot encoded.
    '''
    label_dict = {0: 'N', 1: 'A', 2: 'O', 3: '~'}
    return label_dict[label.argmax()] if one_hot else label_dict[label]