from collections import defaultdict

import pandas as pd
import torch.optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

import config
import in_context_models
import plotting
from classical_models import Linear, LinearVariational, NonLinear, NonLinearVariational
import datasets
from config import dataset_size_classical, device


def train_in_context_models(dx, dy, dh, dataset_amount, dataset_size, num_iters):
    losses = ['mle-params', 'mle-dataset', 'forward-kl', 'backward-kl']
    model_specs = [('Linear', {'order': 1}), ('Linear', {'order': 2}), ('NonLinear', {'dh': dh})]

    trained_models = []

    for model_spec in model_specs:
        for loss in losses:
            model = in_context_models.InContextModel(dx, dy, 32, 4, 5, model_spec[0], loss, **model_spec[1])
            dataset = datasets.ContextDataset(dataset_amount, dataset_size, model_spec[0], dx, dy, **model_spec[1])
            model_trained = train(model, dataset, iterations=num_iters, batch_size=100,
                  eval_dataset=dataset)
            trained_models.append((loss, model_trained))

    return trained_models


def train_classical_models(dx, dy, dh, dataset_size, num_iters):
    # Create underlying ground truth models and datasets for training classical models
    # Also will be the evaluation data used later
    linear_datasets = []
    linear_2_datasets = []
    nonlinear_datasets = []

    tries = 10 # How many datasets to test on

    for _i in range(0, tries):
        gt_linear = Linear(dx, dy, order=1)
        ds_linear = datasets.PointDataset(dataset_size, gt_linear)
        linear = (gt_linear, ds_linear, [])

        gt_linear_2 = Linear(dx, dy, order=2)
        ds_linear_2 = datasets.PointDataset(dataset_size, gt_linear_2)
        linear_2 = (gt_linear_2, ds_linear_2, [])

        gt_nonlinear = NonLinear(dx, dy, dh)
        ds_nonlinear = datasets.PointDataset(dataset_size, gt_nonlinear)
        nonlinear = (gt_nonlinear, ds_nonlinear, [])

        linear_datasets.append(linear)
        linear_2_datasets.append(linear_2)
        nonlinear_datasets.append(nonlinear)

        # Train all combinations
        for dataset in [linear, linear_2, nonlinear]:
            for model in [Linear(dx, dy, order=1), Linear(dx, dy, order=2), NonLinear(dx, dy, dh),
                          LinearVariational(dx, dy, order=1), LinearVariational(dx, dy, order=2),
                          NonLinearVariational(dx, dy, dh)]:
                model_trained = train(model, dataset[1], iterations=num_iters, batch_size=100, gt_model=dataset[0])
                dataset[2].append(model_trained)

    return linear_datasets, linear_2_datasets, nonlinear_datasets


def train(model, dataset, iterations, batch_size, eval_dataset=None, gt_model=None, plot=True):
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if eval_dataset is not None:
        eval_dataset = dataset
        eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # lr 0.001 for classical, 0.0001 for context
    loss_fns = {"MSE": torch.nn.MSELoss()}
    tqdm_batch = tqdm(range(iterations), unit="batch", ncols=100, leave=True)
    for it in tqdm_batch:
        loss = train_step(model, optimizer, loss_fns, iter(dataloader), it)
        tqdm_batch.set_postfix({"loss": loss})

        # plotting in case of amortized models
        if plot:
            if it % 10 == 0 and eval_dataset is not None:
                eval_data_batch = next(iter(eval_dataloader))  #TODO fix iter
                model.plot_eval(eval_data_batch, loss_fns)
            # plotting in case of classical models
            elif it % 500 == 0 and gt_model is not None:
                model.plot_eval(gt_model, loss_fns)

    return model


def train_step(model, optimizer, loss_fns, dataloader_it, it):
    model.train()
    model.zero_grad()
    optimizer.zero_grad()

    batch = next(dataloader_it)
    loss = model.compute_loss(batch, loss_fns)
    loss.backward()
    optimizer.step()
    return loss

if __name__ == "__main__":
    linear_datasets, linear_2_datasets, nonlinear_datasets = train_classical_models(dx=1, dy=1, dh=config.dh, dataset_size=dataset_size_classical, num_iters=config.num_iters_classical)
    trained_in_context_models = train_in_context_models(dx=1, dy=1, dh=config.dh, dataset_amount=config.dataset_amount,
                            dataset_size=config.dataset_size_in_context, num_iters=config.num_iters_in_context)

    X = torch.linspace(-5, 5, 128).unsqueeze(1)  # 128 equally spaced evaluation points between -1 and 1 - should we instead take a normally distributed sample here every time?

    mse_results = []
    for model_type in [linear_datasets, linear_2_datasets, nonlinear_datasets]:
        for elem in model_type:
            gt = elem[0]
            classical_models_trained = elem[2]

            Y = gt(X) # ground truth output to compare with

            for i in range(0, len(classical_models_trained)):
                classical_models_trained[i].eval()
                Y_pred = classical_models_trained[i](X)
                mse = torch.mean((Y-Y_pred)**2)
                mse_results.append({'gt': gt._get_name(), 'model_name': classical_models_trained[i]._get_name(), 'mse': mse.item()})

            for trained_in_context_model in trained_in_context_models:
                Y_pred = trained_in_context_model[1].predict(torch.cat((elem[1].X, elem[1].Y), dim=-1).unsqueeze(0), X.unsqueeze(0))
                mse = torch.mean((Y-Y_pred)**2)
                mse_results.append({'gt': gt._get_name(), 'model_name': trained_in_context_model[0]+" "+trained_in_context_model[1].eval_model._get_name(), 'mse': mse.item()})

    df = pd.DataFrame(mse_results)

    # average over similar columns to compute mean performance across datasets
    df_avg = df.groupby(['gt', 'model_name'], as_index=False)['mse'].mean()

    # Save to disk (choose one or both)
    df_avg.to_csv("experiment1_results.csv", index=False)

    # TODO: AVERAGE correctly, right now only the last value is taken


    print(mse_results)