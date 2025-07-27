import torch

import datasets
import seed
from classical_models import NonLinearVariational, NonLinear, Linear, LinearVariational
from main import train, eval_plot
from config import device

seed.set_seed(0)


dx = 1
dy = 1
dh = 10
dorder = 2
lr = 0.001
dataset_size = 100
num_iters = 100000
batch_size=50

nlr_gt = Linear(dx, dy, dorder)
ds = datasets.PointDataset(dataset_size, nlr_gt)
val_ds = datasets.PointDataset(dataset_size, nlr_gt)
val_freq = 100

nlr_variational = LinearVariational(dx, dy, dorder)
nlr_variational_trained = train(nlr_variational, ds, val_ds, val_freq, num_iters, batch_size, lr, use_wandb=False)

X = torch.linspace(-2,2,25).unsqueeze(1).to(device)

Y_pred = nlr_variational_trained(X)

eval_plot("", "", nlr_gt, X, Y_pred)



