import torch

def get_one_hot(label, num_class=4):
    '''
    convert the label to one-hot encoding
    '''
    one_hot = torch.zeros(num_class, dtype=torch.float32)\
        .scatter_(dim=0, index=torch.tensor(label), value=1)
    return one_hot