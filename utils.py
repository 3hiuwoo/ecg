import torch
import numpy as np
import random

def get_cinc2017_class(labels):
    '''
    decode the one-hot encoding to class
    '''
    label_dict = {0: 'N', 1: 'A', 2: 'O', 3: '~'}
    return [label_dict[l.argmax().item()] for l in labels]


def normalize(arr):
    '''
    normalize the array by x = (x - x.min()) / (x.max() - x.min())
    '''
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr

def set_seed(seed):
    '''
    set the seed
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

