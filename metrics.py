import numpy as np
import torch
from scipy.stats import t


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


def kl_mvn(dist0, dist1):
    (m0, S0) = dist0
    (m1, S1) = dist1
    EPS = 1e-6
    d = m0.size(-1)
    S0+=EPS * torch.eye(d, device=S0.device)
    S1+=EPS * torch.eye(d, device=S1.device)

    """
    Computes KL(N(m0, S0) || N(m1, S1)) for full covariance matrices.
    m*: shape (d,), S*: shape (d,d).
    """
    N = m0.shape[0]

    # use slogdet for numerical stability
    sign0, logdet0 = torch.linalg.slogdet(S0)
    sign1, logdet1 = torch.linalg.slogdet(S1)
    assert sign0 > 0 and sign1 > 0, "Covariance matrices must be positive-definite"

    iS1 = torch.linalg.inv(S1)
    diff = m1 - m0

    tr_term = torch.trace(iS1 @ S0)
    quad_term = diff.T @ iS1 @ diff
    det_term = logdet1 - logdet0

    return 0.5 * (tr_term + quad_term + det_term - N)

def se(x):
    return x.std(ddof=1) / np.sqrt(len(x))

def ci95(x):
    n = len(x)
    std = x.std(ddof=1)
    se_val = std / np.sqrt(n)
    t_val = t.ppf(0.975, df=n - 1)
    return t_val * se_val
