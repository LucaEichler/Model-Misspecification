import torch.optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

import in_context_models
import plotting
from classical_models import Linear, LinearVariational, NonLinear, NonLinearVariational
import datasets

config_path = ".\config.yaml"

with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.loader.SafeLoader)

    num_datasets = config.get("num_datasets")
    noise_std = config.get("noise_std")


def train_in_context_models(dx, dy, dh, dataset_size):
    datasets_linear = datasets.ContextDataset(1000, dataset_size, 'Linear', 1, 1, order=1)
    datasets_linear_test = datasets.ContextDataset(1, dataset_size, 'Linear', 1, 1, order=1)
    model_linear = in_context_models.InContextModel(dx, dy, 512, 4, 5, 'Linear', 'forward-kl', order=1)

    datasets_linear2 = datasets.ContextDataset(1000, dataset_size, 'Linear', 1, 1, order=2)
    datasets_linear2_test = datasets.ContextDataset(1, dataset_size, 'Linear', 1, 1, order=2)
    model_linear2 = in_context_models.InContextModel(dx, dy, 512, 4, 5, 'Linear', 'mle-params', order=2)

    datasets_nonlinear = datasets.ContextDataset(1000, dataset_size, 'NonLinear', 1, 1, dh=20)
    datasets_nonlinear_test = datasets.ContextDataset(1, dataset_size, 'NonLinear', 1, 1, dh=20)
    model_nonlinear = in_context_models.InContextModel(dx, dy, 512, 4, 5, 'NonLinear', 'mle-params', dh=20)

    train(model_nonlinear, datasets_nonlinear, iterations=10000, batch_size=100, eval_dataset=datasets_nonlinear_test)

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


def train(model, dataset, iterations, batch_size, eval_dataset=None):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if eval_dataset is not None:
        eval_dataset = dataset
        eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    # weight decay (?)

    loss_fns = {"MSE": torch.nn.MSELoss()}
    tqdm_batch = tqdm(range(iterations), unit="batch", ncols=100, leave=True)
    for it in tqdm_batch:
        loss = train_step(model, optimizer, loss_fns, dataloader, it)
        tqdm_batch.set_postfix({"loss": loss})
        if it % 10 and eval_dataset is not None:
            eval_data_batch = next(iter(eval_dataloader))
            model.plot_eval(eval_data_batch, loss_fns)





def train_step(model, optimizer, loss_fns, dataloader, it):
    model.train()
    model.zero_grad()

    batch = next(iter(dataloader))
    loss, *_ = model.compute_loss(batch, loss_fns)
    loss.backward()
    optimizer.step()
    return loss


# train_classical_models(dx=1, dy=1, dh=100, dataset_size=5000)
train_in_context_models(dx=1, dy=1, dh=100, dataset_size=50)