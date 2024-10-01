import torch
from torch import nn


def ecg_conv_block(in_channels, out_channels, kernel_size,
               num_conv, stride=1, padding='same'):
    '''
    create block with multiple same size conv layers
    '''
    layers = []
    for _ in range(num_conv):
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding))
        layers.append(nn.ReLU())
        in_channels = out_channels
    return nn.Sequential(*layers)

    
def ecg_conv_net(arch, num_conv=2, fc=128, dropout=0.6, num_class=4,
                 mpkernel=8, mpstride=2):
    '''
    generalized structure of the network proposed in the paper
    '''
    in_channel = 1
    conv_blks = []
    for idx, (out_channel, kernel_size) in enumerate(arch):
        conv_blks.append(conv_block(in_channel, out_channel,
                                    kernel_size, num_conv))
        if idx < (len(arch) - 1):
            conv_blks.append(nn.MaxPool1d(kernel_size=mpkernel, stride=mpstride))
        in_channel = out_channel
    conv_blks.append(nn.AdaptiveMaxPool1d(1))
    return nn.Sequential(*conv_blks, nn.Flatten(), 
                         nn.Linear(in_channel, fc), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(fc, fc), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(fc, num_class))
    
        