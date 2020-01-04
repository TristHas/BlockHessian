import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__),"../../src"))
from utils import zero_grad
from activation_stats import first_order_analysis
from block_analysis import block_hessian, eval_loss, update_params
from models import conv_net
from data import gen_cifar10_ds
from losses import LinearClassification

def get_model_ds_loss(inp_dim, hid_dim, out_dim,
                      nlayer, bias, use_bn, mode,
                      nsamp, device, loss_mode='CrossEntropy'):
    
    model =  conv_net(inp_dim, hid_dim, out_dim, nlayer, bias, use_bn, mode).cuda(device)
    ds = gen_cifar10_ds(nsamp, device, download=False)
    
    assert loss_mode in ["CrossEntropy", "Linear"]
    if loss_mode=='CrossEntropy':
        loss_fn = nn.CrossEntropyLoss()
    elif loss_mode=='Linear':
        loss_fn = LinearClassification(out_dim)
        
    return model, ds, loss_fn

def relative_error(a, b, eps=1e-6):
    """
    """
    return abs((a - b) / min(abs(a), abs(b)))

def init_dir(path):
    if not os.path.exists(path):
        os.makedirs(path) 
        
def save_model(model, epoch, save_model_dir):
    torch.save(model.state_dict(), os.path.join(save_model_dir, f"{epoch}.pth"))

def correct(classifier, target):
    return classifier.max(dim = 1)[1] == target

def train_epoch(model, ds, loss_fn, lr):
    zero_grad(model)
    loss = eval_loss(model, ds, loss_fn)

    grads = [x.grad for x in model.parameters()]
    delta = grads
    params = list(model.parameters())
    
    update_params(params, delta, lr)

    model.zero_grad()
    
    acc = correct(model(ds[0][0]), ds[0][1])

    return loss, acc.type(torch.FloatTensor).mean().item()

def run_analysis(model, ds, loss_fn, lr, epochs, valfreq=40, save_model_dir=None):
    """
    """
    if save_model_dir is not None:
        init_dir(save_model_dir)
    val_stats, tr_stats = [],[]

    for epoch in tqdm(range(epochs)):
        if (epoch+1) % valfreq==0:
            H = block_hessian(model, ds, loss_fn, lr)
            delta, fo, ho, fostat = first_order_analysis(model, ds, loss_fn, lr)
            error = relative_error(H.sum().item(), ho)
            val_stats.append((H, delta, fo, ho, error, fostat))
            if save_model_dir is not None:
                save_model(model, epoch, save_model_dir)

        loss, acc = train_epoch(model, ds, loss_fn, lr)
        tr_stats.append((loss, acc))
    return val_stats, tr_stats
    
def unpack_stats(stats):
    return list(zip(*stats))
    
def plot_acc_loss(acc, loss, ax=None):
    """
    """
    if ax is None:
        fig, ax = plt.subplots(1,2, figsize=(12,5))
    ax[1].plot(loss)
    ax[0].plot(acc)
    
def select(stats, column):
    """
    """
    stats = [stat[column] for stat in stats]
    return pd.concat(stats, axis=1).T.set_index(np.arange(len(stats)))

def plot_stats(stats, column, *args, ax=None, **kwargs):
    """
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,8))
    df = select(stats, column)
    df.plot(*args, ax=ax, **kwargs)
    
def summarize_stats(stats):
    """
    """
    fig, ax = plt.subplots(2,2, figsize=(20, 20))
    plot_stats(stats, 'a_l_std', ax=ax[0,0])
    plot_stats(stats, 'a_l_m', ax=ax[0,1])
    plot_stats(stats, 'W_g_std', ax=ax[1,0])
    plot_stats(stats, 'W_std', ax=ax[1,1])
    
def skip_BN_params(BH, bn):
    if bn.endswith("11"):
        BH = [h.cpu()[0::3,0::3] for h in BH]
    elif bn.endswith("10") or  bn.endswith("01"):
        BH = [h.cpu()[0::2,0::2] for h in BH]
    else:
        BH = [h.cpu() for h in BH]
    return BH

def get_delta_fo_ho(data, bn_code=None, BH=None):
    if BH is None:
        BH = skip_BN_params(data["H"], bn_code)
    lw_fo = [np.array(fo.W_g_sqr)*data["lr"] for fo in data["fostat"]]
    lw_ho = [h.numpy().sum(axis=0) for h in BH]
    lw_delta = [fo + ho for fo, ho in zip(lw_fo, lw_ho)]
    delta_fo_ho = np.array((lw_delta, lw_fo, lw_ho), dtype=np.float).transpose(1,2,0)
    return delta_fo_ho
    
def preprocess(data, bn):
    BH = skip_BN_params(data["H"], bn)
    delta_fo_ho = get_delta_fo_ho(data, BH)
    return {"BH": skip_BN_params(data["H"], bn),
            "delta": delta_fo_ho,
            "loss": data["loss"],
            "acc": data["acc"]}

