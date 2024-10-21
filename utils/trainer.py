import torch
from tqdm import tqdm
from torchmetrics import MeanMetric, Accuracy, F1Score, MetricCollection
from .functional import get_device, Visualizer, Logger, get_pseudo_label


def train_epoch_supervised(model, train_iter, optimizer, loss_fn, metrics,
                epoch, epochs, device):
    '''
    train the model for one epoch under supervised learning paradigm
    
    argument ''metrics'' is a tuple of arbitrary metrics defined outside the function, such
    as loss, accuracy, and f1
    '''
    model.train()

    ls, acc, f1 = metrics

    # initialize the progress bar
    tbar = tqdm(train_iter, desc=f"Epoch [{epoch+1}/{epochs}] training")
    for X, y in tbar:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        
        ls.update(loss)
        train_accuracy = acc(pred, y)
        train_f1 = f1(pred, y)
        
        tbar.set_postfix({'loss':f'{loss.item():.5f}',
                          'accuracy':f'{train_accuracy.item():.5f}',
                          'f1':f'{train_f1.item():.5f}'})

    # average across the batches
    train_loss = ls.compute()
    total_train_acc = acc.compute()
    total_train_f1 = f1.compute()
    
    ls.reset()
    acc.reset()
    f1.reset()
    
    return total_train_acc.item(), total_train_f1.item(), train_loss.item()


def valid_epoch_supervised(model, valid_iter, metrics, epoch, epochs, device):
    '''
    validate the model for one epoch under supervised learning paradigm
    
    if epochs is 0, the function is used for testing on the test set
    
    argument ''metrics'' is a tuple of arbitrary metrics defined outside the
    function, such as accuracy and f1
    '''
    model.eval()
    
    acc, f1 = metrics
    
    # check whether it is testing or validating
    if epochs == 0:
        desc = 'Testing'
    else:
        desc = f"Epoch [{epoch+1}/{epochs}] validating"
    with torch.no_grad():
        # initialize the progress bar
        vbar = tqdm(valid_iter, desc=desc)
        for X, y in vbar:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            
            valid_accuracy = acc(pred, y)
            valid_f1 = f1(pred, y)
            
            vbar.set_postfix({'accuracy':f'{valid_accuracy.item():.5f}',
                              'f1':f'{valid_f1.item():.5f}'})
    
    # average across the batches
    total_valid_acc = acc.compute()
    total_valid_f1 = f1.compute()
    
    acc.reset()
    f1.reset()
    
    return total_valid_acc.item(), total_valid_f1.item()
    

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
    
    # training and validating metrics
    train_loss = MeanMetric().to(device)
    train_accuracy = Accuracy(task='multiclass', num_classes=4).to(device)
    train_f1 = F1Score(task='multiclass', num_classes=4).to(device)
    
    valid_accuracy = Accuracy(task='multiclass', num_classes=4).to(device)
    valid_f1 = F1Score(task='multiclass', num_classes=4).to(device)

    # training process
    for epoch in range(start_epoch+1, epochs):
        # train
        train_acc, train_f1, train_loss =\
            train_epoch_supervised(model, train_iter, optimizer, loss_fn,
                            (train_loss, train_accuracy, train_f1), epoch,
                            epochs, device)

        # validate
        valid_acc, valid_f1 = valid_epoch_supervised(model, valid_iter,
                                                     (valid_accuracy, valid_f1),
                                                    epoch, epochs, device)
        
        # log the metrics
        visualizer.write('Loss/Acc/F1', epoch, train_loss=train_loss,
                         train_accuracy=train_acc, train_f1=train_f1,
                         valid_accuracy=valid_acc, valid_f1=valid_f1)
        
        # save the model
        if (epoch+1) % check == 0:
            logger.save(model, optimizer, epoch)

    visualizer.close()
    print('Training finished')
    
    
def train_epoch_predictive(model, train_iter, optimizer, loss_fn, trans, metrics,
                           weights, epoch, epochs, device):
    '''
    train the model for one epoch under predictive pretext task SSL paradigm
    
    there are several classifiers at the end of the model, each of them is used
    to classify its corresponding transformed data, and the loss is the sum of
    the losses of all classifiers, and the metrics are a tuple of dictionaries,
    each of which contain metrics for every branch, so the argument "trans" is
    needed for indexing, which is a list of strings representing the names of
    transformations to be recognized by the classifiers
    '''
    model.train()
    
    # metrics dictionaries
    losses, acces, f1s = metrics
    
    # initialize the progress bar
    tbar = tqdm(train_iter, desc=f"Epoch [{epoch+1}/{epochs}] training")
    for X, _ in tbar:
        X = X.to(device)
        # get labels for each classifier
        y = get_pseudo_label(X.shape[0], num_tran=len(trans)).to(device)
        
        optimizer.zero_grad()
        # let the transformed siganls stack all together along the batch axis
        pred = model(X.view(-1, 1, X.shape[-1]))
        loss = 0
        
        # update each classifier metrics
        for i, tran in enumerate(trans):
            sub_loss = weights[i] * loss_fn(pred[i], y[i])
            loss += sub_loss
            losses[tran].update(sub_loss)
            acces[tran].update(pred[i], y[i])
            f1s[tran].update(pred[i], y[i])
         
        loss.backward()
        optimizer.step()
       
    # average across the batches 
    total_losses = losses.compute()
    total_acces = acces.compute()
    total_f1s = f1s.compute()
    
    total_losses.reset()
    total_acces.reset()
    total_f1s.reset()
    
    return total_losses, total_acces, total_f1s
    
    
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
    
    # dictionary of some metrics for each classifier
    loss = MetricCollection({tran: MeanMetric()
                             for tran in trans_name}).to(device)
    acc = MetricCollection({tran: Accuracy(task='multiclass', num_classes=2)
                                for tran in trans_name}).to(device)
    f1 = MetricCollection({tran: F1Score(task='multiclass', num_classes=2)
                                for tran in trans_name}).to(device)
    
    # weights for each loss of the classifiers
    weights = torch.tensor([0.195, 0.195, 0.195, 0.0125, 0.0125, 0.195, 0.195],
                          dtype=torch.float32).to(device)
    
    # training process
    for epoch in range(start_epoch+1, epochs):
        total_loss, total_acc, total_f1 = train_epoch_predictive(model,
                                        train_iter, optimizer, loss_fn,
                                        trans_name, (loss, acc, f1), weights,
                                        epoch, epochs, device)
        
        # average each metrics across the classifiers and log them
        mean_loss = sum(total_loss.values()) / len(total_loss)
        visualizer.write('Loss', epoch, total=mean_loss, **total_loss)
        
        mean_acc = sum(total_acc.values()) / len(total_acc)
        visualizer.write('Accuracy', epoch, total=mean_acc, **total_acc)
        
        mean_f1 = sum(total_f1.values()) / len(total_f1)
        visualizer.write('F1score', epoch, total=mean_f1, **total_f1)
        
        # save the model whose backbone and classifier are saved separately
        if (epoch+1) % check == 0:
            logger.save(model, optimizer, epoch)
    
    visualizer.close()
    print('Training finished')
    
    
    