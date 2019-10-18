import sys
import torch
import torch.nn as nn
import torchvision
import numpy as np
from collections import namedtuple

sys.path.append("../src")
from models import *

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

def gen_cifar10_ds(nsamp=1000, device=0, DATA_DIR="./data"):
    """
    """
    dataset = cifar10(root=DATA_DIR)
    x = transpose(normalise(pad(dataset['train']['data'], 4)))
    #y = np.expand_dims(dataset['train']['labels'], axis=-1)
    y = dataset['train']['labels']
    
    index = np.random.choice(np.arange(len(x)), nsamp, replace=False)
    x = torch.Tensor(x)[index]
    y = torch.Tensor(y)[index].type(torch.long)
    
    return [(x.cuda(device), y.cuda(device))]

class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x)

class GAPool(nn.Module):
    def forward(self, x):
        return torch.mean(x,(2,3))

class conv_bn(nn.Module):
    def __init__(self, inp, out, bias=False, use_bn=False, mode="linear"):
        """
        """
        super().__init__()
        self.conv = nn.Conv2d(inp, out, kernel_size=3, stride=1, padding=1, bias=bias)
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(out)
        self.act = Activation(mode)
        
    def forward(self, x):
        """
        """
        #return self.act(self.conv(x))
        return self.act(self.bn(self.conv(x))) if self.use_bn else self.act(self.conv(x))
    
class conv_net(nn.Module):
    def __init__(self, inp, hid, out, nlayer, bias=False, use_bn=False, mode="linear"):
        """
        """
        super().__init__()
        self.l1 = conv_bn(inp, hid, bias=bias, use_bn=use_bn, mode=mode)
        self.layers = nn.Sequential(*[conv_bn(hid, hid, bias=bias, use_bn=use_bn, mode=mode) \
                                      for i in range(max(0,nlayer-2))])
        self.GAPool = GAPool()
        #self.flatten = Flatten()
        self.out = FC(hid, out, bias=bias, mode="linear")
        
    def forward(self, x):
        """
        """
        return self.out(self.GAPool(self.layers(self.l1(x))))#self.out(self.flatten(self.GAPool(self.layers(self.l1(x)))))
    
    
    def get_mode(self):
        """
        """
        return next(self._activations()).mode
    
    def set_mode(self, mode):
        """
        """-7.6964e-08,
        for activation in self._activations():
            activation.set_mode(mode)
    
    def _activations(self):
        """
        """
        return filter(lambda x:isinstance(x, Activation), self.modules())
