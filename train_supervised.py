import torch
import transforms
import warnings
from torch import nn
from torch import optim
from datasets import load_cinc2017
from models import SupervisedConv
from trainers import train

if __name__ == '__main__':
    # ignore trivial warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    model_save_path = './model_state_dict/supervised_conv.pth'
    
    train_iter, valid_iter = load_cinc2017(batch_size=128,
                                           transform=transforms.ToTensor())

    model = SupervisedConv()
    
    loss_fn = nn.CrossEntropyLoss()

    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    epochs = 100

    train(model, train_iter, valid_iter, optimizer, loss_fn, epochs,
          model_save_path)

