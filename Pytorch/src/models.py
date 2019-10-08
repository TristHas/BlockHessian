import torch.nn as nn

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
            msk = (x>0).detach()
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
        
    def forward(self, x):
        """
        """
        return self.act(self.fc(x))
        
class MLP(nn.Module):
    def __init__(self, inp, hid, out, nlayer, bias=False, mode="linear"):
        """
        """
        super().__init__()
        self.l1 = FC(inp, hid, bias=bias, mode=mode)
        self.layers = nn.Sequential(*[FC(hid, hid, bias=bias, mode=mode) \
                                      for i in range(max(0,nlayer-2))])
        self.out = FC(hid, out, bias=bias, mode="linear")
        
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