'''
run this script to train a supervised model.
use python train_downstream.py --help/-h to get argument information.
remember to specify the LOGDIR which is used for tensorboard and the
SAVEPATH to save the model, also BBROOT to read the pretrained model.
'''
import utils.transform as transform
import warnings
from torch import nn
from torch import optim
from datasets.load import load_dataset
from models.load import load_model
from utils.trainer import train_supervised
from utils.functional import get_train_options, set_seed, init_model


if __name__ == '__main__':
    # ignore trivial warnings
    warnings.filterwarnings("ignore")
    
    opt = get_train_options(transfer=True)
    
    # by default use a batch size of 128, 100 epochs, and a learning rate of
    # 0.001, and set the random seed to 42, use the cinc2017 dataset, and the
    # conv model
    batch_size = opt.batch_size
    epochs = opt.epochs
    learning_rate = opt.learning_rate
    set_seed(opt.seed)
    load_fn = load_dataset(opt.dataset)
    data_root = opt.dataroot
    model = load_model(opt.model, 'supervised')
    log_dir = opt.logdir
    save_path = opt.savepath
    resume = opt.resume
    root = opt.bbroot
    isAll = opt.all
    
    init_model(model, root)
    
    train_iter, valid_iter, _ = load_fn(batch_size=batch_size, root=data_root,
                                     transform=transform.ToTensor())

    loss_fn = nn.CrossEntropyLoss()

    if isAll:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    train_supervised(model, train_iter, valid_iter, optimizer, loss_fn, epochs,
          log_dir, save_path, resume=resume)