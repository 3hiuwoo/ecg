from torch import nn
from models.base import conv_backbone, classifier


class SupervisedConv(nn.Module):
    '''supervised convolutional neural network
    
    Args: Please refer to the functions' doc used in the implementation
    '''
    def __init__(self, conv_groups=None, num_conv=2, mpkernel=8, mpstride=2,
                 hidden=512, num_class=4, num_fc=2, dropout=0.6):
        super(SupervisedConv, self).__init__()
        if conv_groups is None:
            conv_groups = [(32, 32), (64, 16), (128, 8)]
        self.backbone = conv_backbone(conv_groups, num_conv, mpkernel, mpstride)
        self.classifier = classifier(conv_groups[-1][0], hidden,
                                     num_class, num_fc, dropout)
        
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x