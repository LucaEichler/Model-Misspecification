from collections import defaultdict, deque

import numpy as np
import pandas as pd
import torch.optim
from matplotlib import pyplot as plt
from scipy.stats import t
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ReduceLROnPlateau

import wandb
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

import config
import in_context_models
from classical_models import Linear, LinearVariational, NonLinear, NonLinearVariational
import datasets
from config import dataset_size_classical, device
from early_stopping import EarlyStopping

# warmup multiplier scheduler for first `warmup` steps
def warmup_fn(step):
    if step < 2000:
        return float(step + 1) / float(max(1, 2000))
    return 1.0


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_in_context_models(dx, dy, x_dist, dataset_amount, dataset_size, batch_size, num_iters, noise_std, model_specs):
    losses = ['mle-params', 'mle-dataset', 'forward-kl', 'backward-kl']

    trained_models = []

    for model_spec in model_specs:
        model_spec_training = model_spec[1].copy()  # these 2 lines ensure that the amortized model does not
        model_spec_training.pop('feature_sampling_enabled', None)  # internally sample sparse features as is done for data generation
        for loss in losses:
            model = in_context_models.InContextModel(dx, dy, 256, 4, 4, model_spec[0], loss, **model_spec_training)  #TODO: Convert into config
            dataset = datasets.ContextDataset(dataset_amount, dataset_size, model_spec[0], dx, dy, x_dist, noise_std, **model_spec[1])
            valset = datasets.ContextDataset(1000, dataset_size, model_spec[0], dx, dy, x_dist, noise_std, **model_spec[1])
            model_trained = train(model, dataset, valfreq=500, valset=valset, iterations=num_iters, batch_size=batch_size,
                  lr=config.lr_in_context, use_wandb=config.wandb_enabled)
            trained_models.append((loss, model_trained))

    return trained_models



def train_classical_models(dx, dy, dh, dataset_size, num_iters):
    # Create underlying ground truth models and datasets for training classical models
    # Also will be the evaluation data used later
    linear_datasets = []
    linear_2_datasets = []
    nonlinear_datasets = []

    tries = config.test_trials # How many datasets to test on

    for _i in range(0, tries):
        gt_linear = Linear(dx, dy, order=1)
        ds_linear = datasets.PointDataset(dataset_size, gt_linear)
        val_ds_linear = datasets.PointDataset(dataset_size, gt_linear)
        linear = (gt_linear, ds_linear, [], val_ds_linear)

        gt_linear_2 = Linear(dx, dy, order=2)
        ds_linear_2 = datasets.PointDataset(dataset_size, gt_linear_2)
        val_ds_linear_2 = datasets.PointDataset(dataset_size, gt_linear_2)
        linear_2 = (gt_linear_2, ds_linear_2, [], val_ds_linear_2)

        gt_nonlinear = NonLinear(dx, dy, dh)
        ds_nonlinear = datasets.PointDataset(dataset_size, gt_nonlinear)
        val_ds_nonlinear = datasets.PointDataset(dataset_size, gt_nonlinear)
        nonlinear = (gt_nonlinear, ds_nonlinear, [], val_ds_nonlinear)

        linear_datasets.append(linear)
        linear_2_datasets.append(linear_2)
        nonlinear_datasets.append(nonlinear)

        # Train all combinations
        for dataset in [linear, linear_2, nonlinear]:
            for model in [Linear(dx, dy, order=1), Linear(dx, dy, order=2), NonLinear(dx, dy, dh),
                          LinearVariational(dx, dy, order=1), LinearVariational(dx, dy, order=2),
                          NonLinearVariational(dx, dy, dh)]:
                model_trained = train(model, dataset[1], valset=dataset[3], valfreq=200, iterations=num_iters, batch_size=100, lr=config.lr_classical, use_wandb=config.wandb_enabled)
                dataset[2].append(model_trained)

    return linear_datasets, linear_2_datasets, nonlinear_datasets


def train(model, dataset, valset, valfreq, iterations, batch_size, lr = 0.001, use_wandb = False):
    if isinstance(model, NonLinear): model._init_weights_training() # TODO ensure good weight init for all models, better code

    if use_wandb:
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

    if valset is not None: valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    if config.early_stopping_enabled:
        early_stopping = EarlyStopping(patience=config.early_stopping_patience, min_delta=config.early_stopping_delta)

    if isinstance(model, in_context_models.InContextModel):
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_fn)
        # cosine decay after warmup: we'll step this manually after warmup period
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(config.num_iters_in_context - 2000), eta_min=1e-6)
        plateau_scheduler = None
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.weight_decay_classical)  # TODO weight decay config
        plateau_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, threshold=1e-4)
    tqdm_batch = tqdm(range(iterations), unit="batch", ncols=100, leave=True)
    i = 0
    #recent_val_losses = deque(maxlen=5)  # keep rolling average of last 5 validation losses
    best_val_loss = torch.tensor(float('inf'))
    for it in tqdm_batch:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)  # restart for fresh epoch
            batch = next(data_iter)
        #batch = dataset.get_batch()
        if isinstance(model, in_context_models.InContextModel):
            if i < 2000:
                scheduler = warmup_scheduler
            else:
                scheduler = cosine_scheduler
        else: scheduler = None
        loss = train_step(model, optimizer, batch, scheduler, it)
        if use_wandb:
            wandb.log({"loss": loss.item(), "iteration": it})
        tqdm_batch.set_postfix({"loss": loss.item()})

        if it % valfreq == 0 and valset is not None:
            val_loss = 0.

            with torch.no_grad():
                for batch in valloader:
                    model.eval()
                    val_loss += model.compute_loss(batch)
                if use_wandb:
                    wandb.log({"val_loss": val_loss.item(), "iteration": it})
            #recent_val_losses.append(val_loss)
            #avg_val_loss = sum(recent_val_losses) / len(recent_val_losses)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # save model with the best validation loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if plateau_scheduler is not None:
                plateau_scheduler.step(val_loss)
            if config.early_stopping_enabled and early_stopping(val_loss, best_val_loss):
                break

    wandb.finish()

    # load best model configuration
    model.load_state_dict(best_model_state)

    return model



def train_step(model, optimizer, batch, scheduler, it):
    model.train()
    model.zero_grad()
    optimizer.zero_grad()

    loss = model.compute_loss(batch)
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return loss

def eval_plot(ds_name, model_name, gt, X_eval, Y_pred, Y_pred_cf=None):
    X = torch.linspace(-2, 2, 25).unsqueeze(1).to(device)
    X = torch.cat([X, X, X], dim=-1)

    Y = gt(X)
    plt.figure(figsize=(6, 4))
    plt.scatter(X[:,0].detach().cpu().numpy(), Y.detach().cpu().numpy())
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.scatter(X_eval.detach().cpu().numpy(), Y_pred.detach().cpu().numpy(), color='orange')
    plt.text(0.01, 0.99, ds_name, transform=plt.gca().transAxes,
            fontsize=12, verticalalignment='top', horizontalalignment='left')

    if Y_pred_cf is not None:
        plt.scatter(X[:, 0].detach().cpu().numpy(), Y_pred_cf.detach().cpu().numpy(), color = 'green')

    # Upper right
    plt.text(0.99, 0.99, model_name, transform=plt.gca().transAxes,
            fontsize=12, verticalalignment='top', horizontalalignment='right')
    plt.savefig("./plots/"+model_name+" - "+ds_name)
    plt.show()
    plt.close()

def eval_plot_nn(name, Y_gt, X_eval, Y_pred):

    plt.figure(figsize=(6, 4))
    plt.scatter(X_eval.detach().cpu().numpy(), Y_gt.detach().cpu().numpy())
    plt.xlim(0, 1)
    plt.ylim(Y_gt.min().item(), Y_gt.max().item())
    plt.scatter(X_eval.detach().cpu().numpy(), Y_pred.detach().cpu().numpy(), color='orange')
    plt.text(0.01, 0.99, "nn- "+name, transform=plt.gca().transAxes,
            fontsize=12, verticalalignment='top', horizontalalignment='left')


    plt.savefig("./plots/"+"nn - "+name)
    plt.show()
    plt.close()


def se(x):
    return x.std(ddof=1) / np.sqrt(len(x))

def ci95(x):
    n = len(x)
    std = x.std(ddof=1)
    se_val = std / np.sqrt(n)
    t_val = t.ppf(0.975, df=n - 1)
    return t_val * se_val

if __name__ == "__main__":

    linear_datasets, linear_2_datasets, nonlinear_datasets = train_classical_models(dx=1, dy=1, dh=config.dh, dataset_size=dataset_size_classical, num_iters=config.num_iters_classical)
    trained_in_context_models = train_in_context_models(dx=1, dy=1, x_dist='gaussian', dataset_amount=config.dataset_amount,
                            dataset_size=config.dataset_size_in_context, batch_size=config.batch_size_in_context,  num_iters=config.num_iters_in_context, noise_std=config.noise_std, compute_closed_form_mle=False, model_specs=[('Linear', {'order': 1}), ('Linear', {'order': 2}), ('NonLinear', {'dh': config.dh})])

    #X = torch.linspace(-5, 5, 128).unsqueeze(1)  # 128 equally spaced evaluation points between -1 and 1 - should we instead take a normally distributed sample here every time?
    X = torch.randn(128).unsqueeze(1).to(device)

    mse_results = []
    mse_params_results = []
    for model_type in [linear_datasets, linear_2_datasets, nonlinear_datasets]:
        j=1
        for elem in model_type:
            gt = elem[0]
            classical_models_trained = elem[2]

            Y = gt(X) # ground truth output to compare with

            for i in range(0, len(classical_models_trained)):

                # we only can compare the parameters if there is no misspecification
                if isinstance(classical_models_trained[i], type(gt)):
                    if not isinstance(classical_models_trained[i], Linear) or classical_models_trained[i].order == gt.order:
                        mse_params = ((classical_models_trained[i].get_W() - gt.get_W())**2).mean()
                        mse_params_results.append({'gt': gt._get_name(), 'model_name': classical_models_trained[i]._get_name(),
                                            'mse_params': mse_params.item()})

                classical_models_trained[i].eval()
                Y_pred = classical_models_trained[i](X)
                mse = torch.mean((Y-Y_pred)**2)
                mse_results.append({'gt': gt._get_name(), 'model_name': classical_models_trained[i]._get_name(), 'mse': mse.item()})
                #eval_plot(gt._get_name()+" "+str(j), classical_models_trained[i]._get_name(), gt, X, Y_pred)

            for trained_in_context_model in trained_in_context_models:

                Y_pred, means_pred = trained_in_context_model[1].predict(torch.cat((elem[1].X, elem[1].Y), dim=-1).unsqueeze(0), X.unsqueeze(0))

                if isinstance(trained_in_context_model[1].eval_model, type(gt)):
                    if not isinstance(trained_in_context_model[1].eval_model, Linear) or trained_in_context_model[1].eval_model.order == gt.order:
                        mse_params = ((means_pred.squeeze(0) - gt.get_W())**2).mean()
                        mse_params_results.append({'gt': gt._get_name(), 'model_name': trained_in_context_model[0]+" "+trained_in_context_model[1].eval_model._get_name(), 'mse_params': mse_params.item()})

                #eval_plot(gt._get_name()+" "+str(j), trained_in_context_model[0]+" "+trained_in_context_model[1].eval_model._get_name(), gt, X, Y_pred)

                mse = torch.mean((Y-Y_pred)**2)

                mse_results.append({'gt': gt._get_name(), 'model_name': trained_in_context_model[0]+" "+trained_in_context_model[1].eval_model._get_name(), 'mse': mse.item()})
            j=j+1
    df = pd.DataFrame(mse_results)
    df_params = pd.DataFrame(mse_params_results)

    # average over similar columns to compute mean performance across datasets
    df_avg = df.groupby(['gt', 'model_name'])['mse'].agg(
        mean_mse='mean',
        std_mse='std',
        se=se,
        ci=ci95
    ).reset_index()
    df_avg_params = df_params.groupby(['gt', 'model_name'], as_index=False)['mse_params'].mean()

    # Save to disk
    df_avg.to_csv("experiment1_results.csv", index=False)
    df_avg_params.to_csv("experiment1_params_results.csv", index=False)

