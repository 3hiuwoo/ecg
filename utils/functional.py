import torch
import os
import argparse
import numpy as np
import random
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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
    '''
    initialize the weights of the model
    '''
    if type(m) == nn.Conv1d or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
            
            
def get_pseudo_label(batch_size, num_tran):
    '''
    get the pseudo label for predictive SSL
    
    the pseudo label is a tensor of shape (num_tran, batch_size) to match a
    desired output shape of the model of (num_tran, batch_size, 2), each tensor
    of the dimension 1 corresponds to the desired output of one of the
    classifiers of the multi-head model.
    '''
    pseudo_label = torch.ones((num_tran, num_tran, 2))
    pseudo_label[..., 1] = 0
    for i in range(num_tran):
        pseudo_label[i, i, :] = torch.tensor([0, 1])
    
    return pseudo_label.repeat(1, batch_size, 1).argmax(-1)


def get_options():
    '''
    fetch the arguments from the terminal
    '''
    parser = argparse.ArgumentParser(description='training settings')
    
    parser.add_argument('--batch_size', type=int, default=128,
                        help='training batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='set random seed')
    parser.add_argument('--dataset', type=str, default='cinc2017',
                        help='dataset used for training')
    parser.add_argument('--model', type=str, default='conv',
                        help='model used for training')
    parser.add_argument('--logdir', type=str, default='./runs',
                        help='directory to save tensorboard log files')
    parser.add_argument('--savepath', type=str,
                        default='./model_state_dict/model_tmp.pth',
                        help='path to save the model')
    parser.add_argument('--resume', action='store_true',
                        help='resume training from checkpoint')
    
    return parser.parse_args()
    
                        

class Visualizer:
    '''show the training process in tensorboard and terminal
    
    Args:
        log_dir(str, optional): the directory to save tensorboard log files
    '''
    def __init__(self, log_dir='runs'):
        if os.path.exists(log_dir):
            os.system(f'rm -rf {log_dir}')
        self.writer = SummaryWriter(log_dir)
        
        
    def write(self, step, **kwargs):
        '''
        write the metrics dictionary to tensorboard and terminal
        '''
        metrics = ' '.join([f'{k}: {v:.2f}' for k, v in kwargs.items()])
        tqdm.write(metrics)
        
        self.writer.add_scalars('Loss&Acc', kwargs, step)


    def close(self):
        self.writer.close()