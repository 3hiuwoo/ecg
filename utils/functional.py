import shutil
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


def init_model(model, root=None, test=False):
    '''
    initialize the weights of the model
    '''
    def init_weights(m):
        if type(m) == nn.Conv1d or type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    
    if root:
        weights = torch.load(root, map_location=get_device())
        model.backbone.load_state_dict(weights['backbone_state_dict'])
        if test:
            model.classifier.load_state_dict(weights['classifier_state_dict'])
        else:
            model.classifier.apply(init_weights)
    else:
        model.apply(init_weights)
        
            
            
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


def get_train_options(transfer=False):
    '''
    fetch the arguments from the terminal for training scripts
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
    parser.add_argument('--check', type=int, default=10,
                        help='save the model every CHECK epochs')
    parser.add_argument('--dataset', type=str, default='cinc2017',
                        help='dataset used for training')
    parser.add_argument('--dataroot', type=str, default='training2017',
                        help='path of the dataset')
    parser.add_argument('--model', type=str, default='conv',
                        help='model used for training')
    parser.add_argument('--logdir', type=str, required=True,
                        help='directory to save tensorboard log files')
    parser.add_argument('--savepath', type=str, required=True,
                        help='path to save the model')
    parser.add_argument('--resume', action='store_true',
                        help='resume training from checkpoint')
    if transfer:
        parser.add_argument('--bbroot', type=str, required=True,
                            help='path of pretrained model')
        parser.add_argument('--all', action='store_true',
                            help='unfreeze the backbone')
    
    return parser.parse_args()
    
    
def get_test_options():
    '''
    fetch the arguments from the terminal for testing script
    '''
    parser = argparse.ArgumentParser(description='testing settings')
    
    parser.add_argument('--batch_size', type=int, default=128,
                        help='testing batch size')
    parser.add_argument('--seed', type=int, default=42,
                        help='set random seed')
    parser.add_argument('--dataset', type=str, default='cinc2017',
                        help='dataset used for training')
    parser.add_argument('--dataroot', type=str, default='training2017',
                        help='path of the dataset')
    parser.add_argument('--model', type=str, default='conv',
                        help='model used for training')
    parser.add_argument('--modelroot', type=str, required=True,
                        help='path of the saved model')
    
    return parser.parse_args()
                        

class Visualizer:
    '''show the training process in tensorboard and terminal
    
    Args:
        log_dir(str, optional): the directory to save tensorboard log files
    '''
    def __init__(self, log_dir, overwrite=False):
        if overwrite and os.path.exists(log_dir):
            shutil.rmtree(log_dir)

        self.writer = SummaryWriter(log_dir)
        
        
    def write(self, title, step, **kwargs):
        '''
        write the metrics dictionary to tensorboard and terminal
        '''
        metrics = ' '.join([f'{k}: {v:.2f}' for k, v in kwargs.items()])
        tqdm.write(metrics)
        
        self.writer.add_scalars(title, kwargs, step)


    def close(self):
        self.writer.close()
        
        
class Logger:
    '''
    manage multiple checkpoints
    
    Args:
        log_dir(str): the directory to save the checkpoints
    '''
    def __init__(self, log_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir
        self.name = os.path.basename(log_dir)

    
    def save(self, model, optimizer, epoch):
        checkpoint = {
            'epoch': epoch,
            'backbone_state_dict': model.backbone.state_dict(),
            'classifier_state_dict': model.classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        path = os.path.join(self.log_dir, f'{self.name}_{epoch+1}.pth')
        torch.save(checkpoint, path)
        
        
    def load(self, device=get_device()):
        latest = self._find_latest()
        checkpoint = torch.load(os.path.join(self.log_dir, latest),
                                map_location=device)
        return checkpoint
    
    
    def _find_latest(self):
        '''
        find the latest checkpoint
        '''
        files = os.listdir(self.log_dir)
        latest = max(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return latest
        
        
    