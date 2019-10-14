import torch
from utils import pair_indexes, init_hessian, \
                  clone_model, get_param, \
                  get_delta_params, dot_product
from block_analysis import update_params, eval_loss, higher_orders

def block_hessian_off_diag(model, ds, loss_fn, lr):
    """
        
    """
    base_model = clone_model(model)
    H = init_hessian(base_model)
    
    delta_norm_sq = torch.zeros_like(H)
    
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
        
        delta_norm_sq[i,j] = delta_norm_sq[j,i] = delta[0].norm().item() * delta[1].norm().item()
        
    return H, delta_norm_sq

def block_hessian_diag(model, ds, loss_fn, lr):
    """
        
    """
    base_model = clone_model(model) 
    diagonal = []
    
    delta_norm_sq = []
    
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
        
        delta_norm_sq.append(delta[0].norm().item()**2)
        
    return torch.cat(list(map(lambda x:x.view((1,)), diagonal))), torch.Tensor(delta_norm_sq).to(torch.cuda.current_device())

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
    d, delta_norm_sq_d = block_hessian_diag(model, ds, loss_fn, lr)
    H, delta_norm_sq_H = block_hessian_off_diag(model, ds, loss_fn, lr)
    
    # delta_norm_sq
    delta_norm_sq_H[range(H.shape[0]), range(H.shape[0])] = delta_norm_sq_d
    
    return _merge_blocks(H, d), delta_norm_sq_H

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

def block_hessian_delta_normed(model, ds, loss_fn, lr):
    BH, delta_norm_sq = block_hessian(model, ds, loss_fn, lr)
    return BH/delta_norm_sq

    for i,j in pair_indexes(model):
        BH[i,j] = BH[j,i] = BH[i,j]/delta_norm_sq[i,j]
        
    for i, _ in enumerate(model.parameters()):
        BH[i,i] = BH[i,i]/delta_norm_sq[i,i]
    
    return BH