import torch
import torch.autograd as autograd


def use_cuda():
    return torch.cuda.is_available()
    # return False


def variable(data, *args, **kwargs):
    var = autograd.Variable(data, *args, **kwargs)
    if use_cuda():
        var = var.cuda()
    return var
