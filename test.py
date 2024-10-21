'''
run this script to test a model which has been trained supervisedly, remember
specify the modelroot which is the path of the saved model.
'''
import utils.transform as transform
import warnings
from torchmetrics import Accuracy, F1Score
from datasets.load import load_dataset
from models.load import load_model
from utils.trainer import valid_epoch_supervised
from utils.functional import get_test_options, set_seed, init_model, get_device


if __name__ == '__main__':
    # ignore trivial warnings
    warnings.filterwarnings("ignore")
    
    opt = get_test_options()
    
    # by default use a batch size of 128, set the random seed to 42, use the
    # cinc2017 dataset, and the conv model
    batch_size = opt.batch_size
    set_seed(opt.seed)
    load_fn = load_dataset(opt.dataset)
    data_root = opt.dataroot
    model = load_model(opt.model, 'supervised')
    modelroot = opt.modelroot
    
    init_model(model, modelroot, test=True)
    
    _, _, test_iter = load_fn(batch_size=batch_size, root=data_root,
                              transform=transform.ToTensor())

    acc = Accuracy(task='multiclass', num_classes=4)
    f1 = F1Score(task='multiclass', num_classes=4)
    
    test_acc, test_f1 = valid_epoch_supervised(model, test_iter, (acc, f1), 0,
                                               0, device=get_device())
    
    print(f'test accuracy: {test_acc}, test f1: {test_f1}')