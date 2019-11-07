import torch
import numpy as np

def gen_rnd_ds(inp_dim, inp_mean=0, inp_var=1, 
               target_dim=10, nsamp=1000, device=0):
    """
    """
    mean = torch.randn((1,inp_dim))*inp_mean
    var  = torch.randn((nsamp,inp_dim))*inp_var
    x = mean + var
    y = torch.randint(0, target_dim, (nsamp,)).long()
    return [(x.cuda(device), y.cuda(device))]

import torchvision

cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border), (0, 0)], mode='reflect')

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 

def cifar10(root):
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    return {
        'train': {'data': train_set.data, 'labels': train_set.targets},
        'test': {'data': test_set.data, 'labels': test_set.targets}
    }

def gen_cifar10_ds(nsamp=1000, device=0, DATA_DIR="../../data", split='train'):
    """
    """
    dataset = cifar10(root=DATA_DIR)
    x = transpose(normalise(pad(dataset['train']['data'], 4)))
    #y = np.expand_dims(dataset['train']['labels'], axis=-1)
    y = dataset[split]['labels']
    
    index = np.random.choice(np.arange(len(x)), nsamp, replace=False)
    x = torch.Tensor(x)[index]
    y = torch.Tensor(y)[index].type(torch.long)
    
    return [(x.cuda(device), y.cuda(device))]