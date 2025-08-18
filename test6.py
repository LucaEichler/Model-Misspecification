# Testing for 2nd task
import torch

import datasets
import seed
from classical_models import Linear
import config
from main import train, eval_plot

num_iters=10
dataset_size=1000
from config import device

model = Linear(dx=3, dy=1, order=3, nonlinear_features_enabled=True, feature_sampling_enabled=True)
ds = datasets.PointDataset(dataset_size, model, noise_std=0.5)


#seed.set_seed(0)
for i in range(100):
    model = Linear(dx=3, dy=1, order=3, nonlinear_features_enabled=True, feature_sampling_enabled=True)
    gt_linear = Linear(dx=3, dy=1, order=3, nonlinear_features_enabled=False, feature_sampling_enabled=True)

    ds_linear = datasets.PointDataset(dataset_size, gt_linear)
    val_ds_linear = datasets.PointDataset(dataset_size, gt_linear)

    dataset=(gt_linear, ds_linear, [], val_ds_linear)

    """model_trained = train(model, dataset[1], valset=dataset[3], valfreq=200, iterations=num_iters, batch_size=100,
                          lr=config.lr_classical, use_wandb=config.wandb_enabled)
    """
    X = torch.linspace(-2,2,25).unsqueeze(1).to(device)
    X = torch.cat([X,X,X], dim =-1)
    Y_pred = model(X)

    eval_plot("", "", gt_linear, X[:,0], Y_pred)