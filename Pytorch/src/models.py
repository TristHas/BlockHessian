import math
import torch
import torch.nn as nn

class BN2d_ctrl(nn.Module):
    def __init__(self, num_features, use_bn=True, eps=1e-5):
        super().__init__()
        if isinstance(use_bn, bool):
            use_bn = [use_bn]*4

        self.ctr = use_bn[0]
        self.std = use_bn[1]
        self.scl = use_bn[2]
        self.bias = use_bn[3]
        self.eps = eps
        if self.scl:
            self.bn_weight = nn.Parameter(torch.ones(num_features))
        if self.bias:
            self.bn_bias = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x):
        if self.ctr:
            x = x - x.mean(dim=(0,2,3), keepdim=True)
        if self.std:
            x = x / x.std(dim=(0,2,3), keepdim=True)
        if self.scl:
            x = torch.mul(x, self.bn_weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        if self.bias:
            x = torch.add(x, self.bn_bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        
        return x

class Activation(nn.Module):
    def __init__(self, mode="linear"):
        """
        """
        super().__init__()
        assert mode in ["relu", "linear"]
        self.mode = mode
        self.last_msk = None
        
    def set_mode(self, mode):
        """
        """
        assert mode in ["relu", "replay", "linear"]
        self.mode=mode
        
    def forward(self, x):
        """
        """
        if self.mode=="relu":
            msk = (x>0).detach().to(x.dtype)
            self.last_msk=msk
            return x*msk
        
        elif self.mode=="replay":
            assert self.mode=="relu" and self.last_msk is not None
            return x*self.last_msk
        else:
            return x

class FC(nn.Module):
    def __init__(self, inp, out, bias=False, mode="linear"):
        """
        """
        super().__init__()
        self.fc = nn.Linear(inp, out, bias=bias)
        self.act = Activation(mode)
        self.init_weights()
        
    def forward(self, x):
        """
        """
        return self.act(self.fc(x))
    
    def init_weights(self, init_type="variance"):
        if init_type=="variance":
            var = {"relu":2, "linear":1}[self.act.mode]
            with torch.no_grad():
                self.fc.weight.normal_(std=math.sqrt(var/self.fc.weight.shape[0]))
        else:
            raise NotImplementedError
            
class MLP(nn.Module):
    def __init__(self, inp, hid, out, nlayer, bias=False, mode="linear"):
        """
        """
        super().__init__()
        self.l1 = FC(inp, hid, bias=bias, mode=mode)
        self.layers = nn.Sequential(*[FC(hid, hid, bias=bias, mode=mode) \
                                      for i in range(max(0,nlayer-2))])
        self.out = FC(hid, out, bias=bias, mode="linear")
        for l in filter(lambda x:isinstance(x,FC), self.modules()):
            l.init_weights()
        
    def forward(self, x):
        """
        """
        return self.out(self.layers(self.l1(x)))
    
    def get_mode(self):
        """
        """
        return next(self._activations()).mode
    
    def set_mode(self, mode):
        """
        """
        for activation in self._activations():
            activation.set_mode(mode)
    
    def _activations(self):
        """
        """
        return filter(lambda x:isinstance(x, Activation), self.modules())
    

###
### Cifar CNN
###
class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x)

class GAPool(nn.Module):
    def forward(self, x):
        return torch.mean(x,(2,3))

class conv_bn(nn.Module):
    def __init__(self, inp, out, stride=1, bias=False, use_bn=False, mode="linear"):
        """
        """
        super().__init__()
        self.conv = nn.Conv2d(inp, out, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn = BN2d_ctrl(out, use_bn)
        self.act = Activation(mode)
        torch.nn.init.kaiming_normal_(self.conv.weight, a={"relu":0, "linear":1}[mode])
        
    def forward(self, x):
        """
        """
        return self.act(self.bn(self.conv(x)))
    
class conv_net(nn.Module):
    def __init__(self, inp, hid, out, nlayer, 
                 bias=False, use_bn=False, mode="linear"):
        """
        """
        super().__init__()
        self.l1 = conv_bn(inp, hid, stride=2, bias=bias, use_bn=use_bn, mode=mode)
        self.layers = nn.Sequential(*[conv_bn(hid, hid, stride=1, bias=bias, use_bn=use_bn, mode=mode) \
                                      for i in range(max(0,nlayer-2))])
        self.GAPool = GAPool()
        self.out = FC(hid, out, bias=False, mode="linear")
        
    def forward(self, x):
        """
        """
        return self.out(self.GAPool(self.layers(self.l1(x))))
    
    def get_mode(self):
        """
        """
        return next(self._activations()).mode
    
    def set_mode(self, mode):
        """
        """
        for activation in self._activations():
            activation.set_mode(mode)
    
    def _activations(self):
        """
        """
        return filter(lambda x:isinstance(x, Activation), self.modules())
