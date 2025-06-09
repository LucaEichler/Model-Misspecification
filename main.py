import torch.optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

import plotting
from classical_models import Linear, LinearVariational, NonLinear, NonLinearVariational
import datasets

config_path = ".\config.yaml"

with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.loader.SafeLoader)

    num_datasets = config.get("num_datasets")
    noise_std = config.get("noise_std")


def train_classical_models(dx, dy, dh, dataset_size):
    # Create underlying ground truth models and datasets for training classical models

    gt_linear = Linear(dx, dy, order=1)
    ds_linear = datasets.PointDataset(dataset_size, gt_linear)

    gt_linear_2 = Linear(dx, dy, order=2)
    ds_linear_2 = datasets.PointDataset(dataset_size, gt_linear_2)

    gt_nonlinear = NonLinear(dx, dy, dh)
    ds_nonlinear = datasets.PointDataset(dataset_size, gt_nonlinear)

    # Train all combinations
    for dataset in [ds_linear, ds_linear_2, ds_nonlinear]:
        for model in [Linear(dx, dy, order=1), Linear(dx, dy, order=2), NonLinear(dx, dy, dh),
                      LinearVariational(dx, dy, order=1), LinearVariational(dx, dy, order=2),
                      NonLinearVariational(dx, dy, dh)]:
            train(model, dataset, iterations=10000, batch_size=100)

    return gt_linear, gt_linear_2, gt_nonlinear


def train(model, dataset, iterations, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # weight decay (?)

    loss_fns = {"MSE": torch.nn.MSELoss()}
    tqdm_batch = tqdm(range(iterations), unit="batch", ncols=100, leave=True)
    for it in tqdm_batch:
        loss = train_step(model, optimizer, loss_fns, dataloader, it)
        tqdm_batch.set_postfix({"loss": loss})


def train_step(model, optimizer, loss_fns, dataloader, it):
    model.train()
    model.zero_grad()

    batch = next(iter(dataloader))
    loss = model.compute_loss(batch, loss_fns)
    loss.backward()
    optimizer.step()
    return loss


train_classical_models(dx=1, dy=1, dh=100, dataset_size=5000)
