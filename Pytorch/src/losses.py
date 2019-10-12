import torch
import torch.nn as nn

###
### Classification losses
###
class SquaredClassification(nn.Module):
    def __init__(self, target_dim, 
                 reduce="mean", device=0):
        """
        """
        assert reduce in ["mean", "sum"]
        super().__init__()
        self.lookup = torch.eye(target_dim).cuda(device)
        self.reduce = {"mean":lambda x:x.mean(),
                       "sum": lambda x:x.sum()}[reduce]
        
    def forward(self, out, label):
        """
        """
        label = self.lookup[label.squeeze()]
        return self.reduce((out-label)**2) / 2
    
class LinearClassification(nn.Module):
    def __init__(self, target_dim, label_term=False,
                 reduce="mean", device=0):
        """
        """
        super().__init__()
        self.lookup = torch.eye(target_dim).cuda(device)
        self.reduce = {"mean":lambda x:x.mean(),
                       "sum": lambda x:x.sum()}[reduce]
        self.label_term = label_term
        
    def forward(self, out, label):
        label = self.lookup[label.squeeze()]
        loss  = - out * label
        if self.label_term:
            raise NotImplementedError
        else:
            return self.reduce(loss)
        
###
### Regression losses
###




