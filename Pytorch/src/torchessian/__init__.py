import torch
from . import complete_mode
from . import batch_mode

def batch_hessian_eigen(model, ds, loss_fn, neigen=20):
    T, V = complete_mode.lanczos(model, loss_fn, ds, neigen, 2)
    D, U = torch.eig(T, eigenvectors=True)
    eigval = D[:, 0].cpu() # All eingenvalues are real
    eigvec = U.t().mm(V).cpu()
    return eigval, eigvec