import torch

def gen_rnd_ds(inp_dim, inp_mean=0, inp_var=1, 
               target_dim=10, nsamp=1000, device=0):
    """
    """
    mean = torch.randn((1,inp_dim))*inp_mean
    var  = torch.randn((nsamp,inp_dim))*inp_var
    x = mean + var
    y = torch.randint(0, target_dim, (nsamp,1))
    return [(x.cuda(device),y.cuda(device))]
