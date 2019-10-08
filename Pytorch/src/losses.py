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
        super().__init__()
        self.lookup = torch.eye(target_dim).cuda(device)
        self.reduce = lambda x:x.mean() if reduce == "mean" \
                                        else lambda x:x.sum()
        
    def forward(self, out, label):
        """
        """
        label = self.lookup[label]
        return self.reduce((out-label)**2)
    
class LinearClassification(nn.Module):
    def __init__(self, target_dim, label_term=False,
                 reduce="mean", device=0):
        """
        """
        super().__init__()
        self.lookup = torch.eye(target_dim).cuda(device)
        self.reduce = lambda x:x.mean() if reduce == "mean" \
                                        else lambda x:x.sum()
        self.label_term = label_term
        
    def forward(self, out, label):
        label = self.lookup[label]
        loss  = -2 * out * label
        if self.label_term:
            raise NotImplementedError
        else:
            return self.reduce(loss)
        
###
### Regression losses
###




