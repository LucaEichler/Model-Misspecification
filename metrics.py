import torch


def mse(a, b):
    return torch.mean((a - b) ** 2)

def mse_rel(a, b):
    return torch.mean((a - b) ** 2) / (torch.mean(b ** 2) + 1e-12)

def mse_range(a, b):
    error = mse(a, b)
    range = torch.max(b) - torch.min(b)
    return error / (range ** 2 + 1e-12)
