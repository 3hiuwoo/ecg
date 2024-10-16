import torch
from tqdm import tqdm
from torchmetrics import Accuracy, MetricCollection
from utils import transform
from utils.functional import get_device, init_weights, Visualizer,\
                            get_pseudo_label, Logger


def train_epoch_supervised(model, train_iter, optimizer, loss_fn, metric,
                epoch, epochs, device):
    '''
    train the model for one epoch under supervised learning paradigm
    '''
    train_loss = 0
    
    model.train()
    # initialize the progress bar
    tbar = tqdm(train_iter, desc=f"Epoch [{epoch+1}/{epochs}] training")
    for X, y in tbar:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_accuracy = metric(pred, y)
        
        tbar.set_postfix({'loss':f'{loss.item():.2f}',
                          'accuracy':f'{train_accuracy.item():.2f}'})

    total_train_acc = metric.compute()
    train_loss /= len(train_iter)
    metric.reset()
    
    return total_train_acc.item(), train_loss


def valid_epoch_supervised(model, valid_iter, metric, epoch, epochs, device):
    '''
    validate the model for one epoch under supervised learning paradigm
    '''
    if valid_iter is None:
        return None
    
    model.eval()
    with torch.no_grad():
        # initialize the progress bar
        vbar = tqdm(valid_iter, desc=f"Epoch [{epoch+1}/{epochs}] validating")
        for X, y in vbar:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            valid_accuracy = metric(pred, y)
            
            vbar.set_postfix({'accuracy':f'{valid_accuracy.item():.2f}'})
            
    total_valid_acc = metric.compute()
    metric.reset()
    
    return total_valid_acc.item() 
    

def train_supervised(model, train_iter, valid_iter, optimizer, loss_fn, epochs,
          log_dir, check_path, check=5, resume=False, device=get_device()):
    '''
    the training process for supervised learning paradigm
    '''
    # resume training from the checkpoint
    logger = Logger(check_path)
    if resume:
        checkpoint = logger.load()
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = -1
    
    # for printing metrics of an epoch and plotting on the tensorboard
    visualizer = Visualizer(log_dir=log_dir, overwrite=not resume)
    
    train_accuracy = Accuracy(task='multiclass', num_classes=4).to(device)
    valid_accuracy = Accuracy(task='multiclass', num_classes=4).to(device)

    for epoch in range(start_epoch+1, epochs):
        train_acc, train_loss = train_epoch_supervised(model, train_iter,
                                optimizer, loss_fn, train_accuracy, epoch,
                                epochs, device)

        valid_acc = valid_epoch_supervised(model, valid_iter, valid_accuracy,
                                           epoch, epochs, device)
        
        visualizer.write('Loss&Acc', epoch, train_loss=train_loss,
                         train_accuracy=train_acc, valid_accuracy=valid_acc)
        
        # save the model
        if (epoch+1) % check == 0:
            logger.save(model, optimizer, epoch)

    visualizer.close()
    print('Training finished')
    
    
def train_predictive_epoch(model, train_iter, optimizer, loss_fn, trans, metrics,
                           epoch, epochs, device):
    '''
    train the model for one epoch under predictive pretext task SSL paradigm
    
    there are several classifiers at the end of the model, each of them is used
    to classify its corresponding transformed data, and the loss is the sum of
    the losses of all classifiers, and the metrics are a dictionary of each
    branches' accuracy, so the argument "trans" is needed for indexing.
    '''
    model.train()
    total_loss = 0
    
    # initialize the progress bar
    tbar = tqdm(train_iter, desc=f"Epoch [{epoch+1}/{epochs}] training")
    for X, _ in tbar:
        X = X.to(device)
        y = get_pseudo_label(X.shape[0], num_tran=len(trans)).to(device)
        
        optimizer.zero_grad()
        pred = model(X.view(-1, 1, X.shape[-1]))
        loss = 0
        
        for i, tran in enumerate(trans):
            sub_loss = loss_fn(pred[i], y[i])
            loss += sub_loss
            metrics[tran].update(pred[i], y[i])
            
        loss.backward()
        optimizer.step()

        tbar.set_postfix({'loss':f'{loss.item():.2f}'})
        
        total_loss += loss.item()

    total_loss /= len(train_iter)
    total_metrics = metrics.compute()
    
    metrics.reset()
    
    return total_loss, total_metrics
    
    
def train_predictive(model, train_iter, trans_name, optimizer, loss_fn, epochs,
                     log_dir, check_path, check=5, resume=False,
                     device=get_device()):
    '''
    the training process for predictive pretext task SSL paradigm
    '''
    # resume training from the checkpoint
    logger = Logger(check_path)
    if resume:
        checkpoint = logger.load()
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = -1

    # for printing metrics of an epoch and plotting them on the tensorboard
    visualizer = Visualizer(log_dir=log_dir, overwrite=not resume)
    
    # dictionary of accuracies for each classifier
    metrics = MetricCollection({tran: Accuracy(task='multiclass', num_classes=2)
                                for tran in trans_name}).to(device)
    
    for epoch in range(start_epoch+1, epochs):
        total_loss, total_metrics = train_predictive_epoch(model, train_iter,
                                        optimizer, loss_fn, trans_name, metrics,
                                        epoch, epochs, device)
        
        visualizer.write('Loss&Acc', epoch, loss=total_loss, **total_metrics)
        
        # save the model whose backbone and classifier are saved separately
        if (epoch+1) % check == 0:
            logger.save(model, optimizer, epoch)
    
    visualizer.close()
    print('Training finished')
    
    
    