import sys
import numpy as np
import torch
sys.path.append("../src")

from block_analysis import block_hessian, curvature_effects

def pp(lr, delta, h, H):
    rel = abs((h.item() - H.sum().item()) / min(abs(H.sum().item()), abs(h.item())))
    ratio = h.item() / H.sum().item()
    print(f"LR {lr:.2E} \t || Delta={delta:.2E}\t ||Error={rel:.2E}  \t|| hoe={h.item()} \t|| H={H.sum().item()}\t||ratio={ratio}")

def lr_range(model, ds, loss_fn, start=-8, stop=8, step=1, log_scale=False):
    for lr in range(start, stop, step):
        if log_scale:
            lr = 10**lr
        H = block_hessian(model, ds, loss_fn, lr)
        delta, h = curvature_effects(model, ds, loss_fn, lr)
        pp(lr, delta, h, H)
        
def lr_search(model, ds, loss_fn, start=-8, stop=8, step=1, log_scale=False, print_log=False, th=1.e-4):
    lr_list = []
    error_list = []
    H_list = []
     
    for lr in np.arange(start, stop, step):
        if log_scale:
            lr = 10.**lr
        H = block_hessian(model, ds, loss_fn, lr)
        delta, h = curvature_effects(model, ds, loss_fn, lr)
        if print_log:
            pp(lr, delta, h, H)
        rel = abs((h.item() - H.sum().item()) / min(abs(H.sum().item()), abs(h.item())))
        
        if log_scale:
            lr = np.log10(lr)
        lr_list.append(lr)
        error_list.append(rel)
        H_list.append(H.sum().item())
    
    diff = np.diff(H_list)
    diff = abs(diff)<th
    diff = np.insert(diff, 0, False)
    
    error_list = np.array(error_list)
    _idx = np.argmin(error_list[diff])
    idx = np.where(diff)[0][_idx]
        
    best = lr_list[idx]
    
    if idx==0:
        inf = lr_list[idx]
    else:
        inf = lr_list[idx-1]
    if idx==len(lr_list)-1:
        sup = lr_list[idx]
    else:
        sup = lr_list[idx+1]
        
    return inf, sup, best
    
def lr_calibrate(model, ds, loss_fn, start=-8, stop=8, step=1, log_scale=False, print_log=False, width=1.e-3):
    
    inf, sup, best = lr_search(model, ds, loss_fn, start, stop, step, log_scale, print_log)
    if print_log:
        if log_scale:
            print(10.**inf, 10.**best, 10.**sup)
        else:
            print(inf,best,sup)
    
    while (sup-inf)>width:
        step = (sup-inf)/10
        inf, sup, best = lr_search(model, ds, loss_fn, inf, sup+step, step, log_scale, print_log)
        if print_log:
            if log_scale:
                print(10.**inf, 10.**best, 10.**sup)
            else:
                print(inf,best,sup)
        
    return best if not log_scale else 10.**best