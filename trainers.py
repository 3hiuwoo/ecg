import torch
import transforms
import warnings
from tqdm import tqdm
from torch import nn
from torch import optim
from torchmetrics import Accuracy
from torch.utils.tensorboard import SummaryWriter
from utils import get_device, init_weights


def train_epoch(model, train_iter, optimizer, loss_fn, epoch, epochs, device):
    train_acc = Accuracy(task='multiclass', num_classes=4).to(device)
    train_loss = 0
    
    model.train()
    tbar = tqdm(train_iter, desc=f"Epoch [{epoch+1}/{epochs}] training")
    for X, y in tbar:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_accuracy = train_acc(pred, y)
        
        tbar.set_postfix(loss=loss.item(), accuracy=train_accuracy.item())

    total_train_acc = train_acc.compute()
    train_loss /= len(train_iter)
    
    return total_train_acc, train_loss

def valid_epoch(model, valid_iter, epoch, epochs, device):
    valid_acc = Accuracy(task='multiclass', num_classes=4).to(device)
    
    model.eval()
    with torch.no_grad():
        vbar = tqdm(valid_iter, desc=f"Epoch [{epoch+1}/{epochs}] validating")
        for X, y in vbar:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            valid_accuracy = valid_acc(pred, y)
            
            vbar.set_postfix(accuracy=valid_accuracy.item())
            
    total_valid_acc = valid_acc.compute()
    
    return total_valid_acc


def train(model, train_iter, valid_iter, optimizer, loss_fn, epochs,
          path, check=5, resume=False, device=get_device()):
    if resume:
        checkpoint = torch.load(path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        model.apply(init_weights)
        start_epoch = -1
    
    model.to(device)
    writer = SummaryWriter()
    
    for epoch in range(start_epoch+1, epochs):
        train_acc, train_loss = train_epoch(model, train_iter, optimizer,
                                            loss_fn, epoch, epochs, device)
        valid_acc = valid_epoch(model, valid_iter, epoch, epochs, device)
        
        tqdm.write(f"Epoch {epoch+1} " +
                   f"training loss: {train_loss:.2f} " +
                   f"training accuracy: {train_acc:.2f} " +
                   f"validation accuracy: {valid_acc:.2f}")
        
        writer.add_scalars('Loss&Acc', {'train_loss': train_loss,
                                        'train_acc': train_acc,
                                        'valid_acc': valid_acc}, epoch)
        
        if (epoch+1) % check == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)

       