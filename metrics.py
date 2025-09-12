import torch


def mse(a, b):
    return torch.mean((a - b) ** 2)

def mse_rel(a, b):
    return torch.mean((a - b) ** 2) / (torch.mean(b ** 2) + 1e-12)

def mse_range(a, b):
    error = mse(a, b)
    range = torch.max(b) - torch.min(b)
    return error / (range ** 2 + 1e-12)

def nrmse_range(y_true, y_pred, epsilon=1e-6):
    mse = torch.mean((y_true - y_pred) ** 2)
    rmse = torch.sqrt(mse)
    range_y = torch.max(y_true) - torch.min(y_true)
    return rmse / (range_y + epsilon)