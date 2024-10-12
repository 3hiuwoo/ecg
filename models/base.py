from torch import nn


def conv_block(in_channels, out_channels, kernel_size,
               num_conv, stride=1, padding='same'):
    '''
    create block with multiple same size conv layers
    
    Args:
        num_conv(int): Number of conv layers
        other args: Same as nn.Conv1d
    '''
    layers = []
    for _ in range(num_conv):
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding))
        layers.append(nn.ReLU())
        in_channels = out_channels
    return nn.Sequential(*layers)

    
def conv_backbone(conv_groups, num_conv, mpkernel, mpstride):
    '''
    generalized structure of the backbone network proposed in the paper
    
    Args:
        conv_groups(list): List of tuples containing the number of out_channels
            and kernel_size for each conv block
        num_conv(int): Number of conv layers in each group
        mpkernel(int): Kernel size of the maxpooling layer
        mpstride(int): Stride of the maxpooling layer
    '''
    in_channels = 1
    layers = []
    for idx, (out_channels, kernel_size) in enumerate(conv_groups):
        layers.append(conv_block(in_channels, out_channels,
                                    kernel_size, num_conv))
        if idx < (len(conv_groups) - 1):
            layers.append(nn.MaxPool1d(kernel_size=mpkernel, stride=mpstride))
        in_channels = out_channels
    layers.append(nn.AdaptiveMaxPool1d(1))
    return nn.Sequential(*layers)


def classifier(in_features, out_features, num_class, num_fc, dropout):
    '''
    the classfication head used in both supervised and self-supervised models
    
    Args:
        in_features(int): Number of input features
        out_features(int): Number of hidden units
        num_class(int): Number of classes to be classified
        num_fc(int): Number of fully connected layers
        dropout(float): Dropout rate
    '''
    layers = []
    layers.append(nn.Flatten())
    for _ in range(num_fc):
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout(dropout))
        in_features = out_features
    layers.append(nn.Linear(in_features, num_class))
    return nn.Sequential(*layers)