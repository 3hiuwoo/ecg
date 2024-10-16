from .selfsupervised import SSLConvPred
from .supervised import SupervisedConv
from utils.functional import get_device

def load_model(name, mode):
    '''
    return different model based on the base model name and training paradigm
    '''
    if name == 'conv':
        if mode == 'supervised':
            return SupervisedConv().to(get_device())
        elif mode == 'predictive':
            return SSLConvPred().to(get_device())