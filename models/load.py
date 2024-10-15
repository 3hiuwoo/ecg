from .selfsupervised import SSLConvPred
from .supervised import SupervisedConv


def load_model(name, mode):
    '''
    return different model based on the base model name and training paradigm
    '''
    if name == 'conv':
        if mode == 'supervised':
            return SupervisedConv()
        elif mode == 'predictive':
            return SSLConvPred()