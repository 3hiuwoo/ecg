import random
import torch
import numpy as np
from utils import normalize
from scipy.interpolate import interp1d


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
        sf(int/float): scaling factor
    '''
    def __init__(self, sf):
        self.sf = sf
    
    
    def __call__(self, ecg):
        return ecg * self.sf


class VerticalFlip:
    '''
    negate the signal
    
    Args:
        norm(bool, optional): normalize the signal after negating
    '''
    def __init__(self, norm=False):
        self.norm = norm
    
    
    def __call__(self, ecg):
        return (normalize(-ecg) if self.norm else -ecg)


class HorizontalFlip:
    '''
    invert the signal temporally
    
    note:
        cannot receive tensor as parameter
    '''
    def __init__(self):
        pass
    
    
    def __call__(self, ecg):
        return np.flip(ecg, axis=-1)


class AddNoise:
    '''
    add noise to the signal
    
    Args:
        snr(int/float): signal to noise ratio
        
    note:
        the dtype of the return tensor will be forced to float64
    '''
    def __init__(self, snr):
        self.snr = snr
    
    
    def __call__(self, ecg):
        noise = np.random.normal(0, self._get_std(ecg, self.snr), ecg.shape)
        return ecg + noise
    
    
    def _get_std(self, arr, snr):
        avg_power_signal = (arr ** 2).mean()
        avg_power_noise = 10 ** ((avg_power_signal - snr) / 10)
        return (avg_power_noise ** 0.5)
    
    
class Permute:
    '''
    permute the signal
    
    Args:
        n(int): number of segments to be divided into
        
    note:
        will return ndarray when receive tensor as parameter
    '''
    def __init__(self, n):
        self.n = n
    
    
    def __call__(self, ecg):
        segs = np.array_split(ecg, self.n, axis=1)
        np.random.shuffle(segs)
        return np.concatenate(segs, axis=1)
    
    
class TimeWarp:
    '''
    warp the signal in time
    
    Args:
        n(int): number of segments to be divided into
        sf(int/float): stretch factor(>1) or squeeze factor(<1)
        
    note:
        will return ndarray when receive tensor as parameter
    '''
    def __init__(self, n, sf):
        self.n = n
        self.sf = sf
        
        
    def __call__(self, ecg):
        segs = np.array_split(ecg, self.n, axis=-1)
        choices = np.random.choice(self.n, self.n//2, replace=False)
        choices.sort()
        
        # stretch/squeeze selected signal
        for i in range(self.n):
            if i in choices:
                segs[i] = self._warp(segs[i], self.sf)
            else:
                segs[i] = self._warp(segs[i], 1/self.sf)
                
        warp_ecg = np.concatenate(segs, axis=1)
        if warp_ecg.shape[-1] < ecg.shape[-1]:
            warp_ecg = np.pad(warp_ecg,
                ((0, 0), (0, ecg.shape[-1] - warp_ecg.shape[-1])))
        elif warp_ecg.shape[-1] > ecg.shape[-1]:
            warp_ecg = warp_ecg[:, :ecg.shape[-1]]
        return warp_ecg
        
        
    def _warp(self, ecg, sf):
        x_old = np.linspace(0, 1, ecg.shape[-1])
        x_new = np.linspace(0, 1, int(ecg.shape[-1] * sf))
        f = interp1d(x_old, ecg, axis=-1)
        return f(x_new)
        
        
        