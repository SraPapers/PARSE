import torch
import numpy as np
import functools
import operator
#------------------------Torch Operation ------------------------
def to_tensor(x, ref=None):
    if torch.is_tensor(x):
        return x
    if ref is None:
        return torch.tensor(x, device="cuda", dtype=torch.float64)
    return torch.tensor(x, device=ref.device, dtype=ref.dtype)

def add_torch(a, b):
    if not torch.is_tensor(a):
        a = to_tensor(a, b if torch.is_tensor(b) else None)
    if not torch.is_tensor(b):
        b = to_tensor(b, a)
    return a + b

def sub_torch(a, b):
    if not torch.is_tensor(a):
        a = to_tensor(a, b if torch.is_tensor(b) else None)
    if not torch.is_tensor(b):
        b = to_tensor(b, a)
    return a - b

def mul_torch(a, b):
    if not torch.is_tensor(a):
        a = to_tensor(a, b if torch.is_tensor(b) else None)
    if not torch.is_tensor(b):
        b = to_tensor(b, a)
    return a * b

def protected_div_torch(x1, x2):
    eps = 1e-10
    if not torch.is_tensor(x1):
        x1 = to_tensor(x1, x2 if torch.is_tensor(x2) else None)
    if not torch.is_tensor(x2):
        x2 = to_tensor(x2, x1)
    x2_safe = torch.where(torch.abs(x2) < eps, torch.full_like(x2, eps), x2)
    return x1 / x2_safe
    
def protected_div(x1, x2):
    x2 = np.where(np.abs(x2) < 1e-10, 1e-10, x2)
    return x1 / x2    
    
def linker_add(*args):
    if len(args) == 0:
        return 0
    if len(args) == 1:
        return args[0]
    return functools.reduce(operator.add, args)