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


def from_numpy(ndarray):
    tensor = torch.from_numpy(ndarray).float()
    if use_cuda():
        tensor = tensor.cuda()
    return tensor


def to_tensor(array):
    tensor = torch.Tensor(array).float()
    if use_cuda():
        tensor = tensor.cuda()
    return tensor
