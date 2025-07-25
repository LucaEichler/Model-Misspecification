from collections import defaultdict

import pandas as pd
import torch.optim
from matplotlib import pyplot as plt

import wandb
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

import config
import in_context_models
import plotting
from classical_models import Linear, LinearVariational, NonLinear, NonLinearVariational
import datasets
from config import dataset_size_classical, device
from early_stopping import EarlyStopping


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_in_context_models(dx, dy, dh, dataset_amount, dataset_size, batch_size, num_iters):
    losses = ['mle-params', 'mle-dataset', 'forward-kl', 'backward-kl']
    model_specs = [('Linear', {'order': 1}), ('Linear', {'order': 2}), ('NonLinear', {'dh': dh})]

    trained_models = []

    for model_spec in model_specs:
        for loss in losses:
            model = in_context_models.InContextModel(dx, dy, 32, 4, 5, model_spec[0], loss, **model_spec[1])
            dataset = datasets.ContextDataset(dataset_amount, dataset_size, model_spec[0], dx, dy, **model_spec[1])
            model_trained = train(model, dataset, iterations=num_iters, batch_size=batch_size,
                  eval_dataset=dataset, lr=config.learning_rate)
            trained_models.append((loss, model_trained))

    return trained_models



def train_classical_models(dx, dy, dh, dataset_size, num_iters):
    # Create underlying ground truth models and datasets for training classical models
    # Also will be the evaluation data used later
    linear_datasets = []
    linear_2_datasets = []
    nonlinear_datasets = []

    tries = 5 # How many datasets to test on

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
                model_trained = train(model, dataset[1], iterations=num_iters, batch_size=100, gt_model=dataset[0], lr=config.learning_rate)
                dataset[2].append(model_trained)

    return linear_datasets, linear_2_datasets, nonlinear_datasets


def train(model, dataset, iterations, batch_size, eval_dataset=None, gt_model=None, plot=True, lr = 0.001):
    if config.wandb_enabled:
        wandb.init(
            project=config.wandb_project_name,
            name=config.wandb_exp_name,
            config={
                "model_name": model._get_name(),
                "iterations": iterations,
                "batch_size": batch_size,

            }
        )
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_iter = iter(dataloader)

    early_stopping = EarlyStopping(patience=1000, min_delta=0.)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #TODO: learning rate config
    loss_fns = {"MSE": torch.nn.MSELoss()}
    tqdm_batch = tqdm(range(iterations), unit="batch", ncols=100, leave=True)
    for it in tqdm_batch:
        """try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)  # restart for fresh epoch
            batch = next(data_iter)"""
        batch = dataset.get_batch()
        loss = train_step(model, optimizer, loss_fns, batch, it)
        if config.wandb_enabled:
            wandb.log({"loss": loss.item(), "iteration": it})
        tqdm_batch.set_postfix({"loss": loss.item()})

        if early_stopping(loss, model):
            break

    wandb.finish()

    return model


def train_step(model, optimizer, loss_fns, batch, it):
    model.train()
    model.zero_grad()
    optimizer.zero_grad()

    loss = model.compute_loss(batch, loss_fns)
    loss.backward()
    optimizer.step()
    return loss

def eval_plot(ds_name, model_name, gt, X_eval, Y_pred):
    X = torch.linspace(-2, 2, 25).unsqueeze(1).to(device)
    Y = gt(X)
    plt.figure(figsize=(6, 4))
    plt.scatter(X.detach().cpu().numpy(), Y.detach().cpu().numpy())
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.scatter(X_eval.detach().cpu().numpy(), Y_pred.detach().cpu().numpy(), color='orange')
    plt.text(0.01, 0.99, ds_name, transform=plt.gca().transAxes,
            fontsize=12, verticalalignment='top', horizontalalignment='left')

    # Upper right
    plt.text(0.99, 0.99, model_name, transform=plt.gca().transAxes,
            fontsize=12, verticalalignment='top', horizontalalignment='right')
    plt.savefig("./plots/"+model_name+" - "+ds_name)
    plt.show()
    plt.close()




if __name__ == "__main__":

    linear_datasets, linear_2_datasets, nonlinear_datasets = train_classical_models(dx=1, dy=1, dh=config.dh, dataset_size=dataset_size_classical, num_iters=config.num_iters_classical)
    trained_in_context_models = train_in_context_models(dx=1, dy=1, dh=config.dh, dataset_amount=config.dataset_amount,
                            dataset_size=config.dataset_size_in_context, batch_size=config.batch_size_in_context,  num_iters=config.num_iters_in_context)

    #X = torch.linspace(-5, 5, 128).unsqueeze(1)  # 128 equally spaced evaluation points between -1 and 1 - should we instead take a normally distributed sample here every time?
    X = torch.randn(128).unsqueeze(1).to(device)

    mse_results = []
    for model_type in [linear_datasets, linear_2_datasets, nonlinear_datasets]:
        j=1
        for elem in model_type:
            gt = elem[0]
            classical_models_trained = elem[2]

            Y = gt(X) # ground truth output to compare with

            for i in range(0, len(classical_models_trained)):
                classical_models_trained[i].eval()
                Y_pred = classical_models_trained[i](X)
                Y_pred = torch.randn_like(Y_pred)
                mse = torch.mean((Y-Y_pred)**2)
                mse_results.append({'gt': gt._get_name(), 'model_name': classical_models_trained[i]._get_name(), 'mse': mse.item()})

            for trained_in_context_model in trained_in_context_models:
                Y_pred = trained_in_context_model[1].predict(torch.cat((elem[1].X, elem[1].Y), dim=-1).unsqueeze(0), X.unsqueeze(0))

                eval_plot(gt._get_name()+" "+str(j), trained_in_context_model[0]+" "+trained_in_context_model[1].eval_model._get_name(), gt, X, Y_pred)

                mse = torch.mean((Y-Y_pred)**2)

                mse_results.append({'gt': gt._get_name(), 'model_name': trained_in_context_model[0]+" "+trained_in_context_model[1].eval_model._get_name(), 'mse': mse.item()})
            j=j+1
    df = pd.DataFrame(mse_results)

    # average over similar columns to compute mean performance across datasets
    df_avg = df.groupby(['gt', 'model_name'], as_index=False)['mse'].mean()

    # Save to disk (choose one or both)
    df_avg.to_csv("experiment1_results.csv", index=False)

    print(mse_results)

