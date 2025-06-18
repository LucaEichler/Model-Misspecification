import torch.optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

import in_context_models
import plotting
from classical_models import Linear, LinearVariational, NonLinear, NonLinearVariational
import datasets
from config import dataset_size_classical, device



def train_in_context_models(dx, dy, dh, dataset_size):
    datasets_linear = datasets.ContextDataset(1000, dataset_size, 'Linear', 1, 1, order=1)
    datasets_linear_test = datasets.ContextDataset(1, dataset_size, 'Linear', 1, 1, order=1)
    model_linear = in_context_models.InContextModel(dx, dy, 32, 4, 5, 'Linear', 'backward-kl', order=1)

    datasets_linear2 = datasets.ContextDataset(1000, dataset_size, 'Linear', 1, 1, order=2)
    datasets_linear2_test = datasets.ContextDataset(1, dataset_size, 'Linear', 1, 1, order=2)
    model_linear2 = in_context_models.InContextModel(dx, dy, 32, 4, 5, 'Linear', 'forward-kl', order=2)

    datasets_nonlinear = datasets.ContextDataset(1000, dataset_size, 'NonLinear', 1, 1, dh=20)
    datasets_nonlinear_test = datasets.ContextDataset(1, dataset_size, 'NonLinear', 1, 1, dh=20)
    model_nonlinear = in_context_models.InContextModel(dx, dy, 32, 4, 5, 'NonLinear', 'forward-kl', dh=20)

    train(model_linear2, datasets_linear2, iterations=10000, batch_size=100, eval_dataset=datasets_linear2_test)

def train_classical_models(dx, dy, dh, dataset_size):
    # Create underlying ground truth models and datasets for training classical models

    gt_linear = Linear(dx, dy, order=1)
    ds_linear = datasets.PointDataset(dataset_size, gt_linear)
    linear = (gt_linear, ds_linear)

    gt_linear_2 = Linear(dx, dy, order=2)
    ds_linear_2 = datasets.PointDataset(dataset_size, gt_linear_2)
    linear_2 = (gt_linear_2, ds_linear_2)

    gt_nonlinear = NonLinear(dx, dy, dh)
    ds_nonlinear = datasets.PointDataset(dataset_size, gt_nonlinear)
    nonlinear = (gt_nonlinear, ds_nonlinear)

    # Train all combinations
    for dataset in [linear, linear_2, nonlinear]:
        for model in [Linear(dx, dy, order=1), Linear(dx, dy, order=2), NonLinear(dx, dy, dh),
                      LinearVariational(dx, dy, order=1), LinearVariational(dx, dy, order=2),
                      NonLinearVariational(dx, dy, dh)]:
            train(model, dataset[1], iterations=10000, batch_size=100, gt_model=dataset[0])

    return gt_linear, gt_linear_2, gt_nonlinear


def train(model, dataset, iterations, batch_size, eval_dataset=None, gt_model=None):
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if eval_dataset is not None:
        eval_dataset = dataset
        eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # lr 0.001 for classical, 0.0001 for context
    # weight decay (?)

    loss_fns = {"MSE": torch.nn.MSELoss()}
    tqdm_batch = tqdm(range(iterations), unit="batch", ncols=100, leave=True)
    for it in tqdm_batch:
        loss = train_step(model, optimizer, loss_fns, dataloader, it)
        tqdm_batch.set_postfix({"loss": loss})

        # plotting in case of amortized models
        if it % 10 == 0 and eval_dataset is not None:
            eval_data_batch = next(iter(eval_dataloader))
            model.plot_eval(eval_data_batch, loss_fns)
        # plotting in case of classical models
        elif it % 500 == 0 and gt_model is not None:
            model.plot_eval(gt_model, loss_fns)





def train_step(model, optimizer, loss_fns, dataloader, it):
    model.train()
    model.zero_grad()

    batch = next(iter(dataloader))
    loss, *_ = model.compute_loss(batch, loss_fns)
    loss.backward()
    optimizer.step()
    return loss



# train_classical_models(dx=1, dy=1, dh=10, dataset_size=dataset_size_classical)
train_in_context_models(dx=1, dy=1, dh=100, dataset_size=50)