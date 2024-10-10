import torch
from torch import nn


def ecg_conv_block(in_channels, out_channels, kernel_size,
               num_conv, stride=1, padding='same'):
    '''
    create block with multiple same size conv layers
    '''
    layers = []
    for _ in range(num_conv):
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding))
        layers.append(nn.ReLU())
        in_channels = out_channels
    return nn.Sequential(*layers)

    
def ecg_conv_net(arch, num_conv, fc, dropout, num_class, mpkernel, mpstride):
    '''
    generalized structure of the network proposed in the paper
    '''
    in_channel = 1
    conv_blks = []
    for idx, (out_channel, kernel_size) in enumerate(arch):
        conv_blks.append(ecg_conv_block(in_channel, out_channel,
                                    kernel_size, num_conv))
        if idx < (len(arch) - 1):
            conv_blks.append(nn.MaxPool1d(kernel_size=mpkernel,
                                          stride=mpstride))
        in_channel = out_channel
    conv_blks.append(nn.AdaptiveMaxPool1d(1))
    return nn.Sequential(*conv_blks, nn.Flatten(), 
                         nn.Linear(in_channel, fc), nn.ReLU(),
                         nn.Dropout(dropout),
                         nn.Linear(fc, fc), nn.ReLU(),
                         nn.Dropout(dropout),
                         nn.Linear(fc, num_class))
    
    
class ECGConvNet(nn.Module):
    '''
    The convolutional network used in the paper
    '''
    def __init__(self, arch=None, num_conv=2, fc=128, dropout=0.6, num_class=4,
                 mpkernel=8, mpstride=2):
        super(ECGConvNet, self).__init__()
        if arch is None:
            self.arch = [(32, 32), (64, 16), (128, 8)]
        else:
            self.arch = arch
        self.net = ecg_conv_net(self.arch, num_conv, fc, dropout, num_class,
                                mpkernel, mpstride)
        
    def forward(self, x):
        return self.net(x)