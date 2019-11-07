import torch
from utils import pair_indexes, init_hessian, \
                  clone_model, get_param, \
                  dot_product, zero_grad
    
def higher_orders(loss_t, loss_t1, lr, grad, delta):
    """
    """
    first_order = dot_product(grad, delta)
    return -2*(loss_t - loss_t1 - lr * first_order) / (lr**2)

def eval_loss(model, ds, loss_fn, compute_grad=True):
    """
    """
    loss = 0
    for x,y in ds:
        if compute_grad:
            loss_ = loss_fn(model(x),y)
            loss_.backward()
        else:
            with torch.no_grad():
                loss_ = loss_fn(model(x),y)
        loss+=loss_.item()
    return loss

def get_grad_loss(model, ds, loss_fn, lr):
    zero_grad(model)
    loss = eval_loss(model, ds, loss_fn, True)
    grads = {i:x.grad.clone() for i,x in enumerate(model.parameters())}
    return grads, loss

def get_gnorms(grads, model):
    with torch.no_grad():
        gnorms = torch.cat([(grads[i]**2).sum().view(1,1) for i,_ in enumerate(model.parameters())])
        return gnorms.mm(gnorms.t())

def _merge_blocks(H, d):
    """
        Substract H_{ij} = (H_{ij}-d_{i}-d_{j})/2
        Set H_{ii} = d_{i}
    """
    D = -(d.view(1,-1) + d.view(-1,1))
    H = (H+D)/2
    H[range(H.shape[0]), range(H.shape[0])]=d
    return H 

class Updated_params():
    def __init__(self, params, deltas, lr):
        self.args = params, deltas, lr
        self.params_vals = [p.data.clone() for p in params]

    def __enter__(self):
        params, deltas, lr = self.args
        for param, delta in zip(params, deltas):
            param.data.add_(-lr, delta)

    def __exit__(self, *args):
        params, _, lr = self.args
        params_vals = self.params_vals
        for param, params_val in zip(params, params_vals):
            param.detach_()
            param.set_(params_val)
            
def _block_hessian_diag(model, ds, loss_fn, grads, loss_t, lr):
    """
        
    """
    diagonal = []
    for i,_ in enumerate(model.parameters()):
        pair  = (get_param(model, i), )
        grad  = (grads[i].clone(),)
        delta = grad
        
        with Updated_params(pair, delta, lr):
            loss_t1 = eval_loss(model, ds, loss_fn, False)
            
        h = higher_orders(loss_t, loss_t1, lr, grad, delta)
        diagonal.append(h)
        
    return torch.cat(list(map(lambda x:x.view((1,)), diagonal)))

def _block_hessian_off_diag(model, ds, loss_fn, grads, loss_t, lr):
    """
        
    """
    H = init_hessian(model)    
    for i,j in pair_indexes(model):
        pair  = (get_param(model, i), get_param(model, j))
        grad  = (grads[i].clone(), grads[j].clone())
        delta = grad 
        
        with Updated_params(pair, delta, lr):
            loss_t1 = eval_loss(model, ds, loss_fn, False)
        h = higher_orders(loss_t, loss_t1, lr, grad, delta)
        H[i,j] = H[j,i] = h
    return H

def block_hessian(model, ds, loss_fn, lr):
    """
        Missing merge_DH(D, H)
    """
    model = clone_model(model)
    grads, loss_t = get_grad_loss(model, ds, loss_fn, lr)
    gnorms = get_gnorms(grads, model)
    d = _block_hessian_diag(model, ds, loss_fn, grads, loss_t, lr)
    H = _block_hessian_off_diag(model, ds, loss_fn, grads, loss_t, lr)
    H = _merge_blocks(H, d)
    return H, gnorms

def curvature_effects(model, ds, loss_fn, lr):
    """
        Returns O(lr) terms of the Taylor expansion:
        hoe = L(theta_t1) - L(theta_t) + lr * (delta_theta * grad_theta)
    """    
    model = clone_model(model) 
    # Get loss(t) and gradients
    loss_t = eval_loss(model, ds, loss_fn, True)
    grads = [x.grad for x in model.parameters()]
    delta = grads#get_delta_params(model, grads)
    params = list(model.parameters())
    
    update_params(params, delta, lr)
    loss_t1 = eval_loss(model, ds, loss_fn, False)
    return loss_t - loss_t1, higher_orders(loss_t, loss_t1, lr, grads, delta)

###
###
###
def update_params(params, deltas, lr):
    """
    """
    for param, delta in zip(params, deltas):
        param.data.add_(-lr, delta)

def legacy_block_hessian_off_diag(model, ds, loss_fn, grads, loss_t, lr):
    """
        
    """
    H = init_hessian(model)    
    for i,j in pair_indexes(model):
        # Copy the full model for now.
        # If this is too slow, we should make update_params
        # a context manager instead.
        c_model = clone_model(model)
        pair  = (get_param(c_model, i), get_param(c_model, j))
        grad  = (grads[i].clone(), grads[j].clone())
        # Compute delta_theta (=normalized gradient vector for now)
        # But we need to consider other training algo (momentum, etc.)
        delta = grad #get_delta_params(model, grad) 
        # TODO: context manager
        update_params(pair, delta, lr) 
        loss_t1 = eval_loss(c_model, ds, loss_fn, False)
        h = higher_orders(loss_t, loss_t1, lr, grad, delta)
        H[i,j] = H[j,i] = h
        
    return H

def legacy_block_hessian_diag(model, ds, loss_fn, grads, loss_t, lr):
    """
        
    """
    diagonal = []
    for i,_ in enumerate(model.parameters()):
        c_model = clone_model(model)        
        pair  = (get_param(c_model, i), )
        grad  = (grads[i].clone(),)
        delta = grad#get_delta_params(model, grad)
        
        # TODO: context manager
        update_params(pair, delta, lr)
        loss_t1 = eval_loss(c_model, ds, loss_fn, False)
        h = higher_orders(loss_t, loss_t1, lr, grad, delta)
        diagonal.append(h)
        
    return torch.cat(list(map(lambda x:x.view((1,)), diagonal)))

def legacy_block_hessian(model, ds, loss_fn, lr):
    """
        Missing merge_DH(D, H)
    """
    grads, loss_t = get_grad_loss(model, ds, loss_fn, lr)
    gnorms = get_gnorms(grads, model)
    d = legacy_block_hessian_diag(model, ds, loss_fn, grads, loss_t, lr)
    H = legacy_block_hessian_off_diag(model, ds, loss_fn, grads, loss_t, lr)
    H = _merge_blocks(H, d)
    #gnorms = get_gnorms(grads, model)
    return H, gnorms
