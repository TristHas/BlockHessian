import torch
import numpy as np
import math


def infer_layer_idx(grads):
    """
        WARNING: code duplication.
        Need to refactor
    """
    return torch.cumsum(torch.LongTensor([0] + [g.numel() for g in grads]), 0)

def layer_wise_norm(model, eig):
    """
    """
    idx = infer_layer_idx(model.parameters())
    nlayer = len(idx)-1
    return np.concatenate([layer_norm(eig, idx, i) for i in range(nlayer)]).T

def layer_norm(eig, idx, layer_id):
    """
    """
    return np.linalg.norm(eig[:, idx[layer_id]:idx[layer_id+1]], axis=1)[None,:]

def hessian_matmul(model, loss_function, v, batch):

    model.zero_grad()

    x, y = batch
    E = loss_function(model(x), y)
    v.requires_grad = False

    grad_result = torch.autograd.grad(
                E,
                (p for p in model.parameters() if p.requires_grad),
                create_graph=True
    )
    grad_result = torch.cat(tuple(p.view(1, -1) for p in grad_result), 1)
    grad_result.backward(v.view(1, -1))

    result = torch.cat(
        tuple(p.grad.view(1, -1) for p in model.parameters() if p.requires_grad),
        1
    )

    return result.squeeze(0)


def gaussian_density(x, mean, sig):
    return torch.exp(-(x - mean) ** 2 / (2 * sig)) / (sig * math.sqrt(2 * math.pi))


def F(x, L, W, m):
    SIG = 0.0032 # SIG ** 2 == 1e-5
    result = torch.zeros_like(x)
    for i in range(m):
        result += gaussian_density(x, L[i], SIG) * W[i]

    return result


