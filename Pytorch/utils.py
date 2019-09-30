import torch
import copy

###
### Mappers
###
def param_norm(grad_vec):
    """
    """
    return sum(map(lambda x:(x**2).sum(), grad_vec)).sqrt()
    
def get_delta_params(model, grad_vec):
    """
    """
    norm = param_norm(grad_vec)
    return list(map(lambda x:x/norm, grad_vec))
        
def dot_product(delta, grad):
    """
    """
    return sum(map(lambda x: (x[0]*x[1]).sum(), zip(delta, grad)))


###
### Model
###
def zero_grad(model):
    """
    """
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

def get_param(model, i):
    """
    """
    return list(model.parameters())[i]

def clone_model(model):
    """
        Returns a deep copy of the model
        + (just in case) initialize zero gradients
    """
    model = copy.deepcopy(model)
    zero_grad(model)
    return model

def get_model_device(model):
    """
    """
    return next(model.parameters()).device

def init_hessian(model):
    """
    """
    N = len(list(model.parameters()))
    device = get_model_device(model)
    return torch.zeros((N, N), device=device)

def pair_indexes(model):
    """
    """
    for i,_ in enumerate(model.parameters()):
        for j,_ in enumerate(model.parameters()):
            if i<j:
                yield i,j
                
