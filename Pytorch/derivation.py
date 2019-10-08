import torch
from utils import zero_grad

def gradient(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False):
    '''
        Compute the gradient of `outputs` with respect to `inputs`
        gradient(x.sum(), x)
        gradient((x * y).sum(), [x, y])
    '''
    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)
    grads = torch.autograd.grad(outputs, inputs, grad_outputs,
                                allow_unused=True,
                                retain_graph=retain_graph,
                                create_graph=create_graph)
    grads = [x if x is not None else torch.zeros_like(y) for x, y in zip(grads, inputs)]
    return torch.cat([x.contiguous().view(-1) for x in grads])


def jacobian(outputs, inputs, create_graph=False):
    '''
        Compute the Jacobian of `outputs` with respect to `inputs`
        jacobian(x, x)
        jacobian(x * y, [x, y])
        jacobian([x * y, x.sqrt()], [x, y])
    '''
    if torch.is_tensor(outputs):
        outputs = [outputs]
    else:
        outputs = list(outputs)

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    jac = []
    for output in outputs:
        output_flat = output.view(-1)
        output_grad = torch.zeros_like(output_flat)
        for i in range(len(output_flat)):
            output_grad[i] = 1
            jac += [gradient(output_flat, inputs, output_grad, True, create_graph)]
            output_grad[i] = 0
    return torch.stack(jac)

def hessian(output, inputs, out=None, allow_unused=False, create_graph=False):
    '''
        Compute the Hessian of `output` with respect to `inputs`
        hessian((x * y).sum(), [x, y])
    '''
    assert output.ndimension() == 0

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    n = sum(p.numel() for p in inputs)
    
    if out is None:
        out = output.new_zeros(n, n)

    ai = 0
    for i, inp in enumerate(inputs):
        [grad] = torch.autograd.grad(output, inp, create_graph=True, allow_unused=allow_unused)
        grad = torch.zeros_like(inp) if grad is None else grad
        grad = grad.contiguous().view(-1)

        for j in range(inp.numel()):
            if grad[j].requires_grad:
                row = gradient(grad[j], inputs[i:], retain_graph=True, create_graph=create_graph)[j:]
            else:
                row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)

            out[ai, ai:].add_(row.type_as(out))  # ai's row
            if ai + 1 < n:
                out[ai + 1:, ai].add_(row[1:].type_as(out))  # ai's column
            del row
            ai += 1
        del grad

    return out

def infer_layer_idx(grads):
    """
    """
    return torch.cumsum(torch.LongTensor([0] + [g.numel() for g in grads]), 0)

def block_norm(H, deltas):
    """
    """
    N = len(deltas)
    idx = infer_layer_idx(deltas)
    BHN = torch.zeros(N,N)
    
    for i in range(N):
        for j in range(N):
            h = H[idx[i]:idx[i+1], idx[j]:idx[j+1]]
            BHN[i,j]=torch.norm(h)
            
    return BHN

def block_multiply(H, deltas):
    """
    """
    N = len(deltas)
    idx = infer_layer_idx(deltas)
    DHD = torch.zeros(N,N)
    
    for i in range(N):
        for j in range(N):
            h = H[idx[i]:idx[i+1], idx[j]:idx[j+1]]
            a,b = deltas[i], deltas[j]
            dhd = a.view(1,-1).mm(h).mm(b.view(-1,1)).item()
            DHD[i,j]=dhd
    return DHD

def get_hessian_grad_block(model, ds, loss_fn):
    """
    """
    x,y = ds[0]

    # Get gradients
    zero_grad(model)
    loss_fn(model(x),y).backward()
    grads = [x.grad.clone().view(-1) for i,x in enumerate(model.parameters())]
    grad_norms = [torch.norm(g) for g in grads]

    zero_grad(model)
    out = loss_fn(model(x),y)
    H = hessian(out, model.parameters())
    DHD = block_multiply(H, grads)
    return H, grads, DHD