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


def KL_diag_gauss(dist1, dist2):
    """
    Calculate the KL divergence between two diagonal gaussian distributions
    """
    means1, vars1 = dist1
    means2, vars2 = dist2

    KL_div = 0.5 * torch.sum(
        (vars1 / vars2)
        + ((means2 - means1) ** 2) / vars2
        - 1
        + torch.log(vars2 / vars1),
        dim=-1  # sum over feature dimension
    )
    return KL_div