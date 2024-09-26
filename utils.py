import torch
import os
import numpy as np
import pandas as pd
from wfdb import rdrecord


def get_one_hot(label, num_class=4):
    '''
    convert the label to one-hot encoding
    '''
    one_hot = torch.zeros(num_class, dtype=torch.float32)\
        .scatter_(dim=0, index=torch.tensor(label), value=1)
    return one_hot

def get_label(ann_dir):
    '''
    read the label file and return the label dataframe
    '''
    labels = pd.read_csv(ann_dir, header=None)
    labels.columns = ['head', 'label']
    label_dict = {'N': 0, 'A': 1, 'O': 2, '~': 3}
    labels.replace(label_dict, inplace=True)
    return labels

def get_data(ann_dir, ecg_dir, seg, sf):
    '''
    read all the ecg signals and segment them, then merge with the labels
    '''
    npo = seg * sf

    labels = get_label(ann_dir)

    with open(os.path.join(ecg_dir, 'RECORDS'), 'r') as f:
        heads = f.readlines()
    heads = [head.strip() for head in heads]

    signals = {}
    for head in heads:
        record = rdrecord(os.path.join(ecg_dir, head))
        signal = record.p_signal
        signal = np.array(signal).reshape(-1)
        signals[head] = signal

    # segement the signals into a dict
    data = {}
    for k, v in signals.items():
        cnt = len(v) // npo
        if cnt == 0:
            continue
        data[k] = [v[cnt*3000:(cnt+1)*3000] for cnt in range(cnt)]

    # merge by pandas
    df = pd.DataFrame.from_dict(data, orient='index').reset_index()
    df = df.melt(id_vars='index', value_name='signal').sort_values(
        by=['index', 'variable']).drop('variable', axis=1).dropna()
    flatten = df['signal'].apply(pd.Series)
    df = pd.concat([df.drop('signal', axis=1), flatten], axis=1)
    df.reset_index(drop=True, inplace=True)
    df.columns = ['head'] + list(range(npo))

    seg_data = df.merge(labels, on='head')

    return seg_data

def get_data_csv(ecg_dir='training2017', ann_dir='data/REFERENCE-v3.csv',
                 des=os.getcwd, seg=10, sf=300):
    '''
    store the data to a csv file
    '''
    df = get_data(ann_dir, ecg_dir, seg, sf)
    df.to_csv(os.path.join(des, f'train_{seg}s.csv'), index=False)