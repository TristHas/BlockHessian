import torch.nn as nn
import torch.functional as F

def base_debug():
    ### Init
    model =  nn.Linear(inp_dim, out_dim, bias=False).cuda(device)
    model_ = clone_model(model)

    x,y = gen_rnd_ds(inp_dim, inp_mean, inp_var, 
                   out_dim, nsamp, device)[0]

    loss_fn = LinearLoss(out_dim)
    lr = 10
    
    ### Manual
    w = next(model.parameters())
    loss = loss_fn(model(x), y)

    loss.backward()
    grad = w.grad.clone()
    delta = grad/F.norm(grad)

    w.detach_()
    w.add_(-lr, delta)

    loss1 = loss_fn(model(x), y)
    higher_orders(loss, loss1, lr, grad, (delta,)).item()
    
    ### Our code
    ds = [(x,y)]
    pair = (get_param(model_, 0),)
    loss_ = get_loss_gradient(model_, ds, loss_fn)

    grad_ = list(map(lambda x:x.grad.clone(), pair))
    delta_ = get_delta_params(model_, grad_)

    update_params(pair, delta_, lr)

    loss_t1 = get_loss_gradient(model_, ds, loss_fn)

    higher_orders(loss_, loss_t1, lr, grad_, delta_).item()
