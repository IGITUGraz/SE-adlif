import torch

# SLAYER surrogate gradient function
def SLAYER(x, alpha, c):
    return c * alpha / (2 * torch.exp(x.abs() * alpha))