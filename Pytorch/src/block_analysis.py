import torch
from utils import pair_indexes, init_hessian, \
                  dot_product, zero_grad, \
                  Updated_params, eval_loss, \
                  higher_orders, base_get_params,\
                  copy_grad, clone_model
#get_param, \
    
def block_hessian(model, ds, loss_fn, lr, get_params=base_get_params):
    """
        Missing merge_DH(D, H)
    """
    
    model = clone_model(model)
    loss_t = eval_loss(model, ds, loss_fn, True)
    grads = copy_grad(model, get_params)
    
    
    d = _block_hessian_diag(model, ds, loss_fn, grads, loss_t, lr, get_params=get_params)
    H = _block_hessian_off_diag(model, ds, loss_fn, grads, loss_t, lr, get_params=get_params)
    H = _merge_blocks(H, d)
    return H

def _block_hessian_diag(model, ds, loss_fn, grads, loss_t, lr,
                        get_params=base_get_params):
    """
    """
    diagonal = []
    for key, params in get_params(model).items():  
        grad  = [x.clone() for x in grads[key]]
        delta = grad
        
        with Updated_params(params, delta, lr):
            loss_t1 = eval_loss(model, ds, loss_fn, False)
            
        h = higher_orders(loss_t, loss_t1, lr, grad, delta)
        diagonal.append(h)
    return torch.cat(list(map(lambda x:x.view((1,)), diagonal)))

def _block_hessian_off_diag(model, ds, loss_fn, grads, loss_t, lr,
                           get_params=base_get_params):
    """
        
    """
    H = init_hessian(model, get_params)  
    params = get_params(model)
    for (i,key1), (j,key2) in pair_indexes(params):
        pair  = params[key1] + params[key2] 
        grad  = [x.clone() for x in grads[key1] + grads[key2]]
        delta = grad 
        
        with Updated_params(pair, delta, lr):
            loss_t1 = eval_loss(model, ds, loss_fn, False)
        h = higher_orders(loss_t, loss_t1, lr, grad, delta)
        H[i,j] = H[j,i] = h
    return H

def _merge_blocks(H, d):
    """
        Substract H_{ij} = (H_{ij}-d_{i}-d_{j})/2
        Set H_{ii} = d_{i}
    """
    D = -(d.view(1,-1) + d.view(-1,1))
    H = (H+D)/2
    H[range(H.shape[0]), range(H.shape[0])]=d
    return H 