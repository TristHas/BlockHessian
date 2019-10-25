import torch
import matplotlib.pyplot as plt
from .utils import layer_wise_norm

def plot_eigenval(val, ax=plt):
    """
    """
    ax.scatter(torch.arange(val.shape[0]), torch.sort(val, descending=True)[0])
        
def plot_eigenvec(vec, model, ax=plt):
    """
    """
    ax.imshow(layer_wise_norm(model, vec))
    
def plot_leading_eigen(e_val, e_vec, model, N=20, ax=None):
    """
    """
    if ax is None:
        fig, ax = plt.subplots(1,2, figsize=(20,10))
    plot_eigenval(e_val[:N], ax[0])
    #ax[1].imshow(layer_wise_norm(model, e_vec)[:10])
    plot_eigenvec(e_vec[:N], model, ax[1])