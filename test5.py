import torch

import datasets
import seed
from classical_models import NonLinearVariational, NonLinear, Linear, LinearVariational
from main import train, eval_plot, eval_plot_nn
from config import device

seed.set_seed(1)


dx = 1
dy = 1
dh = 10
dorder = 2
lr = 0.001
weight_decay = 1e-4
dataset_size = 128
num_iters = 100000
batch_size=100
x_dist = 'gaussian'

nlr_gt = Linear(dx, dy, 2)
ds = datasets.PointDataset(dataset_size, nlr_gt, x_dist=x_dist)
val_ds = datasets.PointDataset(dataset_size, nlr_gt, x_dist=x_dist)
val_freq = 500

nlr_variational = NonLinearVariational(dx, dy, dh=100)
nlr_variational_trained = train(nlr_variational, ds, val_ds, val_freq, num_iters, batch_size, lr, use_wandb=False, save_path=None, weight_decay=weight_decay)

X = torch.linspace(-2,2,25).unsqueeze(1).to(device)

Y_pred = nlr_variational_trained(X)
Y_gt = nlr_gt(X)

eval_plot_nn("", Y_gt, X, Y_pred)



