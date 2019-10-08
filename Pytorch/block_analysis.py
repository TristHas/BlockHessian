import torch
from utils import pair_indexes, init_hessian, \
                  clone_model, get_param, \
                  get_delta_params, dot_product

def update_params(params, deltas, lr):
    """
    """
    for param, delta in zip(params, deltas):
        param.data.add_(-lr, delta)
    
def higher_orders(loss_t, loss_t1, lr, grad, delta):
    """
    """
    first_order = dot_product(grad, delta)
    return -2*(loss_t - loss_t1 - lr*first_order) / (lr**2)

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

def block_hessian_off_diag(model, ds, loss_fn, lr):
    """
        
    """
    base_model = clone_model(model)
    H = init_hessian(base_model)
    # Get loss(t) and gradients
    loss_t = eval_loss(base_model, ds, loss_fn, True)
    grads = {i:x.grad for i,x in enumerate(base_model.parameters())}
    
    for i,j in pair_indexes(base_model):
        # Copy the full model for now.
        # If this is too slow, we should make update_params
        # a context manager instead.
        model = clone_model(base_model)
        pair  = (get_param(model, i), get_param(model, j))
        grad  = (grads[i].clone(), grads[j].clone())
        # Compute delta_theta (=normalized gradient vector for now)
        # But we need to consider other training algo (momentum, etc.)
        delta = grad#get_delta_params(model, grad) 
        # Possible context manager
        update_params(pair, delta, lr) 
        loss_t1 = eval_loss(model, ds, loss_fn, False)
        
        h = higher_orders(loss_t, loss_t1, lr, grad, delta)
        H[i,j] = H[j,i] = h
    return H

def block_hessian_diag(model, ds, loss_fn, lr):
    """
        
    """
    base_model = clone_model(model) 
    diagonal = []
    
    # Get loss(t) and gradients
    loss_t = eval_loss(base_model, ds, loss_fn, True)
    grads = {i:x.grad for i,x in enumerate(base_model.parameters())}

    for i,_ in enumerate(base_model.parameters()):
        model = clone_model(base_model)        
        pair  = (get_param(model, i), )
        grad  = (grads[i].clone(),)
        delta = grad#get_delta_params(model, grad)
        
        update_params(pair, delta, lr)
        loss_t1 = eval_loss(model, ds, loss_fn, False)
        h = higher_orders(loss_t, loss_t1, lr, grad, delta)
        diagonal.append(h)
        
    return torch.cat(list(map(lambda x:x.view((1,)), diagonal)))

def _merge_blocks(H, d):
    """
        Substract H_{ij} = (H_{ij}-d_{i}-d_{j})/2
        Set H_{ii} = d_{i}
    """
    D = -(d.view(1,-1) + d.view(-1,1))
    H = (H+D)/2
    H[range(H.shape[0]), range(H.shape[0])]=d
    return H 

def block_hessian(model, ds, loss_fn, lr):
    """
        Missing merge_DH(D, H)
    """
    d = block_hessian_diag(model, ds, loss_fn, lr)
    H = block_hessian_off_diag(model, ds, loss_fn, lr)
    return _merge_blocks(H, d)

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