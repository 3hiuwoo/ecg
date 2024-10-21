import torch
from torch import nn
from .base import conv_backbone, classifier


class SSLConvPred(nn.Module):
    '''self-supervised convolutional neural network
    
    Args:
        num_trans(int): number of transformations applied on the input signal
        other args: please refer to the functions' doc used in the implementation
    '''
    def __init__(self, conv_groups=None, num_trans=6, num_conv=2, mpkernel=8,
                 mpstride=2, hidden=128, num_fc=2, dropout=0.6):
        super(SSLConvPred, self).__init__()
        if conv_groups is None:
            conv_groups = [(32, 32), (64, 16), (128, 8)]
        self.backbone = conv_backbone(conv_groups, num_conv, mpkernel, mpstride)
        # classifiers for each transformation
        self.classifier = nn.ModuleDict(
            {f'head_{i}': classifier(conv_groups[-1][0], hidden,
                                     num_class=2, num_fc=num_fc,
                                     dropout=dropout)
             for i in range(num_trans+1)})
        
        
    def forward(self, x):
        outputs = []
        x = self.backbone(x)
        for i in range(len(self.classifier)):
            outputs.append(self.classifier[f'head_{i}'](x))
        return torch.stack(outputs, dim=0)