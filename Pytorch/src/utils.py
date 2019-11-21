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
    return grad_vec#list(map(lambda x:x/norm, grad_vec))
        
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

class Updated_params():
    def __init__(self, params, deltas, lr):
        self.args = params, deltas, lr
        self.params_vals = [p.data.clone() for p in params]

    def __enter__(self):
        params, deltas, lr = self.args
        with torch.no_grad():
            for param, delta in zip(params, deltas):
                param.data.add_(-lr, delta)

    def __exit__(self, *args):
        params, _, lr = self.args
        params_vals = self.params_vals
        with torch.no_grad():
            for param, params_val in zip(params, params_vals):
                param.set_(params_val)

def eval_loss(model, ds, loss_fn, compute_grad=True):
    """
        WARNING: first_order analysis incompatibility in multi-batch
    """
    loss = 0
    for x,y in ds:
        if compute_grad:
            x.requires_grad = True
            loss_ = loss_fn(model(x),y)
            loss_.backward()
        else:
            with torch.no_grad():
                loss_ = loss_fn(model(x),y)
        loss+=loss_.item()
    return loss

def higher_orders(loss_t, loss_t1, lr, grad, delta):
    """
    """
    fo = first_order(grad, delta, lr)
    return (loss_t - loss_t1 - fo) 

def first_order(grad, delta, lr):
    """
    """
    return lr * dot_product(grad, delta)