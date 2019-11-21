import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import zero_grad

DEFAULT_LTYPE = {nn.Conv2d, nn.Linear}

filter_mod_name = lambda x: str(x.__class__).split(".")[-1].split("'")[0]
_chdim  = lambda x: tuple(set(range(x.dim())) - {1})
_tonp   = lambda x: x.detach().cpu().numpy()

###
### Analysis functions
###
ch_mean = lambda x: _tonp(x.mean(_chdim(x)))
ch_std  = lambda x: _tonp(x.std(_chdim(x)))

def get_param_stats(param):
    """
    """
    mean = param.mean().item()
    std = param.std().item()
    g_mean = None if param.grad is None else param.grad.mean().item()
    g_std = None if param.grad is None else param.grad.std().item()
    g_sqr = None if param.grad is None else (param.grad**2).sum().item()
    return mean, std, g_mean, g_std, g_sqr

def get_activ_stats(inp):
    """
    """
    if inp is not None:
        mean, chmean = inp.mean().item(), ch_mean(inp)
        std, chstd  = inp.std().item(), ch_std(inp)
    else:
        mean, chmean, std, chstd = [],[],[],[]
    return mean, std, chmean, chstd

def process_stats(stats):
    """
    """
    df = pd.DataFrame(stats).T
    df = df.set_index(df[1] + "_" + df[0].apply(str))
    df = df[df.columns[2:]]
    df.columns = ["a_l_m", "a_l_std", "A_c_m", "A_c_std", 
                  "g_l_m", "g_l_std", "G_c_m", "G_c_std", 
                  "W_m", "W_std", "W_g_m", "W_g_std", "W_g_sqr"]
    df["a_c_m_std"] = df["A_c_m"].apply(np.std)
    df["a_c_std_m"] = df["A_c_std"].apply(np.mean)
    df["g_c_m_std"] = df["G_c_m"].apply(np.std)
    df["g_c_std_m"] = df["G_c_std"].apply(np.mean)
    return df

###
### Logic
###
def first_order_analysis(model, inp=None, lbl=None, loss_fn=None, ltype=DEFAULT_LTYPE):
    """
    """
    zero_grad(model)
    activ_record = extract_activ_stats(model, inp, lbl, loss_fn, ltype)
    param_record = extract_param_stats(model) 
    record = merge_records(activ_record, param_record)
    return process_stats(record)

def extract_activ_stats(model, inp=None, lbl=None, loss_fn=None, ltype=DEFAULT_LTYPE):
    """
    """
    records = {}
    inp.requires_grad = True
    layers = select_layers(model, ltype)
    fwd_hr = register_fwd_hooks(model, records, layers)
    bwd_hr = register_bwd_hooks(model, records, layers)
    loss = run_model(model, inp, lbl, loss_fn)
    [h.remove() for h in fwd_hr + bwd_hr]
    return records
    
def extract_param_stats(model, ltype=DEFAULT_LTYPE):
    """
    """
    record = {}
    layers = select_layers(model, ltype)
    for layer in layers:
        params = layer.parameters()
        record[layer] = get_param_stats(layer.weight)
    return record

def merge_records(activ, params):
    """
    """
    assert activ.keys() == params.keys()
    return {k:activ[k] + params[k] for k in activ.keys()}

def select_layers(model, ltype=DEFAULT_LTYPE):
    """
    """
    check_ltype = lambda x: type(x) in ltype 
    return list(filter(check_ltype, model.modules()))    
    
def run_model(model, inp=None, lbl=None, loss_fn=None):
    """
    """
    
    out = model(inp)
    if loss_fn is not None:
        if lbl is not None:
            loss = loss_fn(out, lbl)
        else:
            loss = loss_fn(out)
    else:
        loss = (out**2).mean()
    loss.backward()
    return loss.item()

def register_fwd_hooks(model, records, layers):
    """
    """
    def hook(module, inp, out):
        idx = len(records)
        ltype = filter_mod_name(module)
        stats = get_activ_stats(inp[0])
        records[module] = (idx, ltype) + stats
    return [layer.register_forward_hook(hook) for layer in layers]

def register_bwd_hooks(model, records, layers):
    """
    """
    def hook(module, g_inp, g_out):
        stats = get_activ_stats(g_inp[0])
        records[module] += stats
    return [layer.register_backward_hook(hook) for layer in layers]

###
### Plotters
###
def plot_activ(stats, ax=None):
    if ax is None:
        fig, ax = plt.subplots(2,2, figsize=(10,10))
        
    stats.a_l_m.plot(ax=ax[0,0])
    stats.a_l_std.plot(ax=ax[0,0])
    ax[0,0].legend()

    stats.g_l_m.plot(ax=ax[0,1])
    stats.g_l_std.plot(ax=ax[0,1])
    ax[0,1].legend()

    stats.a_c_m_std.plot(ax=ax[1,0])
    stats.a_c_std_m.plot(ax=ax[1,0])
    ax[1,0].legend()

    stats.g_c_m_std.plot(ax=ax[1,1])
    stats.g_c_std_m.plot(ax=ax[1,1])
    ax[1,1].legend()
    
    
def plot_param(stats, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(20,10))
    stats.W_m.plot(ax=ax[0])
    stats.W_std.plot(ax=ax[0])
    ax[0].legend()
    
    stats.W_g_m.plot(ax=ax[1])
    stats.W_g_std.plot(ax=ax[1])
    ax[1].legend()