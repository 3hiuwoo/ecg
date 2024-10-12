import torch
from torch import nn
import numpy as np
import random

def get_device():
    '''
    find the available device
    '''
    return ("cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu")
    

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


def init_weights(m):
        if type(m) == nn.Conv1d or type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)