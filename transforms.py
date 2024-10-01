import torch
import numpy as np
from utils import normalize

def to_one_hot(label, num_class=4):
    '''
    convert the label to one-hot encoding
    '''
    one_hot = torch.zeros(num_class, dtype=torch.float32)\
        .scatter_(dim=0, index=torch.tensor(label), value=1)
    return one_hot


class ToTensor:
    '''
    convert ndarrays to Tensors
    '''
    def __init__(self):
        pass
    
    
    def __call__(self, ecg):
        return torch.from_numpy(ecg)
    
    
class Normalize:
    '''
    normalize the signal by x = (x - x.min()) / (x.max() - x.min())
    '''
    def __init__(self):
        pass
    
    
    def __call__(self, ecg):
        return normalize(ecg)
    
    
class Scale:
    '''
    scale the signal
    
    Args:
        factor(int/float): scaling factor
    '''
    def __init__(self, factor):
        self.factor = factor
    
    
    def __call__(self, ecg):
        return ecg * self.factor


class VerticalFlip:
    '''
    negate the signal
    
    Args:
        norm(bool, optional): normalize the signal after negating
    '''
    def __init__(self norm=False):
        self.norm = norm
    
    
    def __call__(self, ecg):
        return (normalize(-ecg) if self.norm else -ecg)


class HorizontalFlip:
    '''
    invert the signal temporally
    '''
    def __init__(self):
        pass
    
    
    def __call__(self, ecg):
        if isinstance(ecg, torch.Tensor):
            return torch.flip(ecg, [1])
        else:
            return np.flip(ecg, axis=1)


