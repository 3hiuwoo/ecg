import random
import torch
from torch import nn
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt


def get_device():
    '''
    find the available device
    '''
    return ("cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu")


def train_epoch(model, dataloader, optimizer, loss_fn,
                metric, device=get_device()):
    '''
    train the model for one epoch
    '''
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric.update(pred, y)
        total_loss += loss.item()
    
        if batch % 40 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
    train_acc = metric.compute()
    train_loss = total_loss / len(dataloader)
    print(f"Train accuracy: {train_acc:.2f}")
    print(f"Train loss: {train_loss:.2f}")
    metric.reset()
    
    return train_loss, train_acc


def valid_epoch(model, dataloader, metric, device=get_device()):
    '''
    validate the model for one epoch
    '''
    model.eval()
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            metric.update(pred, y)
            
    valid_acc = metric.compute()
    print(f"Valid accuracy: {valid_acc:.2f}")
    metric.reset()

    return valid_acc
    
    
def my_train(model, train_iter, valid_iter, optimizer, loss_fn,
             epochs=1, device=get_device()):
    '''
    simple function for training
    '''
    def init_weights(m):
        if type(m) == nn.Conv1d or type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)
    
    print('Training on', device)
    train_metric = torchmetrics.classification.BinaryAccuracy().to(device)
    valid_metric = torchmetrics.classification.BinaryAccuracy().to(device)
    model.to(device)
    
    for i in range(epochs):
        print(f"----------Epoch {i+1}----------")
        train_epoch(model, train_iter, optimizer, loss_fn, train_metric, device)
        valid_epoch(model, valid_iter, valid_metric, device)
        
    print("Done!")