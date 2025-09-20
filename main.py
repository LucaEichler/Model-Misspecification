import os
from collections import defaultdict, deque
import uuid

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

def save_checkpoint(model, optimizer, scheduler, iters, loss, wandb_id, filename):
    checkpoint = {
        "iters": iters,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loss": loss
    }
    if wandb_id is not None:
        checkpoint["wandb_id"] = wandb_id
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename, device):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iters = checkpoint["iters"]
    loss = checkpoint["loss"]
    return model, optimizer, iters, loss


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_in_context_models(dx, dy, transformer_arch, x_dist, train_specs, model_specs, losses, early_stopping_params, noise_std=None, ds=None, save_path="./default_save_path", save_all = True):

    trained_models = []

    for model_spec in model_specs:
        model_spec_training = model_spec[1].copy()  # these 2 lines ensure that the amortized model does not
        model_spec_training.pop('feature_sampling_enabled', None)  # internally sample sparse features as is done for data generation
        if ds is not None:
            dataset = ds[0]
            valset = ds[1]
        else:
            dataset = datasets.ContextDataset(train_specs['dataset_amount'], train_specs['dataset_size'], model_spec[0], dx, dy, x_dist, noise_std, **model_spec[1])
            valset = datasets.ContextDataset(train_specs['valset_size'], train_specs['dataset_size'], model_spec[0], dx, dy, x_dist, noise_std, **model_spec[1]) #TODO valset size in config
        for loss in losses:
            model = in_context_models.InContextModel(dx, dy, transformer_arch, model_spec[0], loss, train_specs['normalize'], **model_spec_training)  #TODO: Convert into config
            model_name = loss + " " + model.eval_model._get_name()
            model_path = save_path+"/models/"+model_name

            if os.path.exists(model_path+".pt"):    # this path only exists when the train loop for a model was fully finished
                checkpoint = torch.load(model_path+".pt", map_location=config.device)   # in this case, we load the model and skip training
                model.load_state_dict(checkpoint["model_state_dict"])
                model_trained = model
            else:
                os.makedirs(model_path + "/", exist_ok=True)
                model_trained = train(model, dataset, valfreq=500, valset=valset, iterations=train_specs['num_iters'], batch_size=train_specs['batch_size'],
                      lr=train_specs['lr'], weight_decay=train_specs['weight_decay'], early_stopping_params=early_stopping_params, use_wandb=config.wandb_enabled, min_lr = train_specs['min_lr'], save_path=model_path, wandb_name=model_name, save_all=save_all)
            trained_models.append((loss, model_trained))

    return trained_models


def train_classical_models(dx, dy, dh, x_dist, specs, tries):
    dataset_size = specs['dataset_size']

    # Create underlying ground truth models and datasets for training classical models
    # Also will be the evaluation data used later
    linear_datasets = []
    linear_2_datasets = []
    nonlinear_datasets = []


    for _i in range(0, tries):
        gt_linear = Linear(dx, dy, order=1)
        ds_linear = datasets.PointDataset(dataset_size, gt_linear, x_dist)
        val_ds_linear = datasets.PointDataset(dataset_size, gt_linear, x_dist)
        linear = (gt_linear, ds_linear, [], val_ds_linear)

        gt_linear_2 = Linear(dx, dy, order=2)
        ds_linear_2 = datasets.PointDataset(dataset_size, gt_linear_2, x_dist)
        val_ds_linear_2 = datasets.PointDataset(dataset_size, gt_linear_2, x_dist)
        linear_2 = (gt_linear_2, ds_linear_2, [], val_ds_linear_2)

        gt_nonlinear = NonLinear(dx, dy, dh)
        ds_nonlinear = datasets.PointDataset(dataset_size, gt_nonlinear, x_dist)
        val_ds_nonlinear = datasets.PointDataset(dataset_size, gt_nonlinear, x_dist)
        nonlinear = (gt_nonlinear, ds_nonlinear, [], val_ds_nonlinear)

        linear_datasets.append(linear)
        linear_2_datasets.append(linear_2)
        nonlinear_datasets.append(nonlinear)

        # Train all combinations
        for dataset in [linear, linear_2, nonlinear]:
            for model in [LinearVariational(dx, dy, order=1), LinearVariational(dx, dy, order=2),
                          NonLinearVariational(dx, dy, dh), Linear(dx, dy, order=1), Linear(dx, dy, order=2), NonLinear(dx, dy, dh),
                           ]:
                model_trained = train(model, dataset[1], valset=dataset[3], valfreq=1000, weight_decay=specs['weight_decay'], iterations=specs['num_iters'], batch_size=specs['batch_size'], lr=specs['lr'], use_wandb=False, save_path=None, early_stopping_params=specs['early_stopping_params'])
                dataset[2].append(model_trained)

    return linear_datasets, linear_2_datasets, nonlinear_datasets

def load_latest_checkpoint(model, optimizer=None, scheduler=None, dir="defaultdir"):
    # find all checkpoint files
    if dir is None: return 0, None
    checkpoints = [f for f in os.listdir(dir) if f.endswith(".pt")]
    if not checkpoints:
        return 0, None

    # get latest model by iteration number of file
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    latest = checkpoints[-1]
    path = os.path.join(dir, latest)
    print(f"Loading checkpoint: {path}")

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if "wandb_id" in checkpoint:
        wandb_id = checkpoint["wandb_id"]
    else: wandb_id = None
    return checkpoint.get("iters"), wandb_id

def train(model, dataset, valset, valfreq, iterations, batch_size, lr, weight_decay, save_path, use_wandb = False, early_stopping_params=config.early_stopping_params, min_lr=1e-5, wandb_name="Some Experient",
          save_all=True):
    if isinstance(model, NonLinear): model._init_weights_training() # TODO ensure good weight init for all models, better code

    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_iter = iter(dataloader)

    if valset is not None: valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    if early_stopping_params['early_stopping_enabled']:
        early_stopping = EarlyStopping(patience=early_stopping_params['patience'], min_delta=early_stopping_params['min_delta'])

    if isinstance(model, in_context_models.InContextModel):
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        warmup_iters = 2000
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0 / warmup_iters, end_factor=1.0, total_iters=warmup_iters
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(iterations - warmup_iters), eta_min=min_lr
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_iters]
        )
        plateau_scheduler = None
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # TODO weight decay config
        plateau_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, threshold=1e-4)
        scheduler = None

    start_iter, wandb_id = load_latest_checkpoint(model, optimizer, scheduler, save_path)
    min_save_iters = 0

    if use_wandb:
        if wandb_id is None:
            wandb_id = str(uuid.uuid4())
        wandb.init(
            project=config.wandb_project_name,
            name=wandb_name,
            config={
                "model_name": model._get_name(),
                "iterations": iterations,
                "batch_size": batch_size,
            },
            id=wandb_id,
            resume="allow"
        )

    tqdm_batch = tqdm(range(start_iter, iterations), unit="batch", ncols=100, leave=True, initial=start_iter)
    best_val_loss = torch.tensor(float('inf'))
    for it in tqdm_batch:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)  # restart for fresh epoch
            batch = next(data_iter)

        loss = train_step(model, optimizer, batch, scheduler, it+start_iter)
        if use_wandb:
            wandb.log({"loss": loss.item(), "iteration": it+start_iter})
        tqdm_batch.set_postfix({"loss": loss.item()}) #TODO maybe switch to val loss

        if (it+start_iter) % valfreq == 0 and valset is not None:
            val_loss, val_i = 0., 0.

            with torch.no_grad():
                for batch in valloader:
                    model.eval()
                    val_loss += model.compute_loss(batch)
                    val_i += 1.
                val_loss = val_loss/val_i
                if use_wandb:
                    wandb.log({"val_loss": val_loss.item(), "iteration": it+start_iter})
            if val_loss < best_val_loss:    # execute if validation loss reaches a new best
                best_val_loss = val_loss
                if it+start_iter >= min_save_iters and it != 0:  # save if some minimum threshold of iterations has been reached
                    if save_path is not None:
                        path = save_path+"/_"+str(it+start_iter)+".pt" if save_all else save_path + "/best_1.pt"
                        save_checkpoint(model, optimizer, scheduler, it+start_iter, best_val_loss, wandb_id, path)
            if (it + start_iter) % 50000 == 0:  # every 50000 steps, save a backup file so that training may be continued from that point
                if save_path is not None: save_checkpoint(model, optimizer, scheduler, it + start_iter, best_val_loss, wandb_id,
                                save_path + "/backup_0" + ".pt")

            if plateau_scheduler is not None:
                plateau_scheduler.step(val_loss)
            if early_stopping_params['early_stopping_enabled'] and early_stopping(val_loss, best_val_loss):
                break
    wandb.finish()

    # if load best = True, the best model wrt. val loss is loaded and returned. if load best = False, the model state after the final iteration is returned
    if early_stopping_params['load_best']: load_latest_checkpoint(model, optimizer, scheduler, save_path)
    # save the final model in a different location than the checkpoints, to be accessed for evaluation
    if save_path is not None: save_checkpoint(model, optimizer, scheduler, iterations, best_val_loss, wandb_id, save_path+".pt")

    return model



def train_step(model, optimizer, batch, scheduler, it):
    model.train()
    model.zero_grad()
    optimizer.zero_grad()

    loss = model.compute_loss(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #TODO test out if can be removed
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return loss

def eval_plot(ds_name, model_name, gt, X_eval, Y_pred, Y_pred_cf=None, savepath=None):
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
    plt.savefig(savepath+"/plots/"+model_name+" - "+ds_name)
    #plt.show()
    plt.close()

def eval_plot_nn(name, Y_gt, X_eval, Y_pred):

    plt.figure(figsize=(6, 4))
    plt.scatter(X_eval.detach().cpu().numpy(), Y_gt.detach().cpu().numpy())
    plt.xlim(0, 1)
    plt.xlim(-5, 5)
    plt.ylim(Y_gt.min().item(), Y_gt.max().item())
    plt.scatter(X_eval.detach().cpu().numpy(), Y_pred.detach().cpu().numpy(), color='orange')
    plt.text(0.01, 0.99, "nn- "+name, transform=plt.gca().transAxes,
            fontsize=12, verticalalignment='top', horizontalalignment='left')


    plt.savefig("./plots/"+"nn - "+name)
    plt.show()
    plt.close()



