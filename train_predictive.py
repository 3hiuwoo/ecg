'''
run this script to train a model under predictive pretext task SSL paradigm.
use python train_predictive.py --help/-h to get argument information.
remember to specify the ''logdir'' which is used for tensorboard and the
''savepath'' to save the model.
'''
import utils.transform as transform
import warnings
from torch import nn
from torch import optim
from torchvision.transforms import Compose
from datasets.load import load_dataset
from models.load import load_model
from utils.trainer import train_predictive
from utils.functional import get_train_options, set_seed, init_model


if __name__ == '__main__':
    # ignore trivial warnings
    warnings.filterwarnings("ignore")

    opt = get_train_options()
    
    # by default use a batch size of 128, 100 epochs, and a learning rate of
    # 0.001, and set the random seed to 42, use the cinc2017 dataset, and the
    # conv model
    batch_size = opt.batch_size
    epochs = opt.epochs
    learning_rate = opt.learning_rate
    set_seed(opt.seed)
    load_fn = load_dataset(opt.dataset)
    data_root = opt.dataroot
    model = load_model(opt.model, 'predictive')
    log_dir = opt.logdir
    save_path = opt.savepath
    check = opt.check
    resume = opt.resume
    
    tranls = [
        transform.AddNoise(15), transform.Scale(0.9),
        transform.VerticalFlip(), transform.HorizontalFlip(),
        transform.Permute(20), transform.TimeWarp(9, 1.05)
    ]
    trans = Compose([transform.ToGroup(tranls), transform.ToTensor()])

    # no need for validation set
    train_iter, _, _ = load_fn(batch_size=batch_size, root=data_root,
                            transform=trans)
 
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    acc_names = ['origin', 'noise', 'scale', 'vflip', 'hflip', 'permute', 'warp']

    if not resume:
        init_model(model)
    
    train_predictive(model, train_iter, acc_names, optimizer, loss_fn, epochs,
                     log_dir, save_path, check, resume=resume)
    
    
    
    
    