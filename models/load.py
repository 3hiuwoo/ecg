from .selfsupervised import SSLConvPred
from .supervised import SupervisedConv


def load_model(name, mode):
    if name == 'conv':
        if mode == 'supervised':
            return SupervisedConv()
        elif mode == 'predictive':
            return SSLConvPred()