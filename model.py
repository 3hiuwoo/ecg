import torch
from torch import nn


def conv_block(in_channels, out_channels, kernel_size,
               num_conv, stride=1, padding='same'):
    '''
    create block with multiple same size conv layers
    '''
    layers = []
    for _ in range(num_conv):
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size,
                                padding=padding))
        layers.append(nn.ReLU())
        in_channels = out_channels
    return nn.Sequential(*layers)

    
def conv_net(arch, num_conv=2):
    '''
    network used in paper
    '''
    in_channels = 1
    conv_blks = []
    for idx, (out_channel, kernel_size) in enumerate(arch):
        conv_blks.append(conv_block(in_channels, out_channel,
                                    kernel_size, num_conv))
        if idx < (len(arch) - 1):
            conv_blks.append(nn.MaxPool1d(kernel_size=8, stride=2))
        in_channels = out_channel
    conv_blks.append(nn.AdaptiveMaxPool1d(1))
    return nn.Sequential(*conv_blks, nn.Flatten(), 
                         nn.Linear(in_channels, 128), nn.ReLU(), nn.Dropout(0.6),
                         nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.6),
                         nn.Linear(128, 4))
    
        