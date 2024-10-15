from .cinc2017 import load_cinc2017

def load_dataset(name):
    # return different dataset loader selected by the name
    if name == 'cinc2017':
        return load_cinc2017