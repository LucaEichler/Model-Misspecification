import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from plotly.subplots import make_subplots
from scipy.stats import norm

import config
import datasets
import in_context_models
import metrics
from classical_models import LinearVariational, Linear
import plotly.graph_objects as go
import numpy as np
from main import train
import seaborn as sns


import wandb
import matplotlib.pyplot as plt


def plot_two_metrics_seaborn(model_name, save_folder="./plots", figsize=(22, 7), dpi=150, color='blue'):
 # orange mle params, red mle-dataset , green forward-kl
    # Set Seaborn style
    sns.set(style="whitegrid", context="talk")

    # Paths
    path_loss = f"./exp2_loss_plots/exp2 {model_name} loss.csv"
    path_val = f"./exp2_loss_plots/exp2 {model_name} val.csv"

    df_loss = pd.read_csv(path_loss)
    df_val = pd.read_csv(path_val)

    # Create folder if needed
    os.makedirs(save_folder, exist_ok=True)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Training Loss
    sns.lineplot(x='Step', y=f"{model_name} - loss", data=df_loss, ax=axes[0], color=color, linewidth=2)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")

    # Validation Loss
    sns.lineplot(x='Step', y=f"{model_name} - val_loss", data=df_val, ax=axes[1], color=color, linewidth=2)
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Validation Loss")

    """y_min = min(df_loss[f"{model_name} - loss"].min(), df_val[f"{model_name} - val_loss"].min())
    y_max = max(df_loss[f"{model_name} - loss"].max(), df_val[f"{model_name} - val_loss"].max())
    axes[0].set_ylim(y_min, y_max)
    axes[1].set_ylim(y_min, y_max)"""

    # Overall title
    #plt.suptitle(model_name, fontsize=16)
    plt.tight_layout(w_pad=10.0)

    # Save
    save_path = os.path.join(save_folder, f"exp2 {model_name}.pdf")
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    print(f"Saved plot to {save_path}")

#plot_two_metrics_seaborn("nop_Linear", color= 'black')
#plot_two_metrics_seaborn("nop_Polynomial", color ='black')
#plot_two_metrics_seaborn("nop_Nonlinear", color ='black')


def plot_two_metrics(model_name):
    path_loss = "./exp1_loss_plots/exp1 "+model_name+" loss.csv"
    path_val = "./exp1_loss_plots/exp1 "+model_name+" val.csv"
    df_loss = pd.read_csv(path_loss)
    df_val = pd.read_csv(path_val)

    fig, axes = plt.subplots(1, 2, figsize=(10,4))

    axes[0].plot(df_loss['Step'], df_loss[model_name+" - loss"])
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Iteration")

    axes[1].plot(df_val['Step'], df_val[model_name + " - val_loss"])
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Iteration")

    plt.tight_layout()
    plt.savefig("file.pdf")
    plt.close()


def plot_switched_data():
    switched = np.load("switched.npy")
    not_switched = np.load("not_switched.npy")
    plt.figure(figsize=(6, 4))
    plt.scatter(switched[:, 0], switched[:, 1], color = 'orange', label='Nonlinear Training Data')
    plt.scatter(not_switched[:, 0], not_switched[:, 1], color = 'blue', label='Polynomial Training Data')
    plt.legend()
    plt.xlabel("MSE (Closed Form)")
    plt.ylabel("Parameter Loss")
    plt.ylim(-0.1, 1.)
    plt.grid()
    plt.savefig("switch_data.pdf")
    plt.show()

def plot_3d_surfaces(model1, model2, W1, W2, model_name="model", ds_name="ds"):
    surfaces = []
    surfaces2 = []
    N = 10

    x_min, y_min = -10., -10.
    x_max, y_max = 10., 10.
    x_points, y_points = 20, 20
    x = torch.linspace(x_min, x_max, x_points)
    y = torch.linspace(y_min, y_max, y_points)
    X1, X2 = torch.meshgrid(x, y, indexing='ij')

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        horizontal_spacing=0.05
    )

    # Compute global z-range across all frames
    z_min, z_max = float('inf'), -float('inf')
    for k in range(N):
        x3 = float(k)*2-10.
        points = torch.stack([X1.reshape(-1), X2.reshape(-1), x3*torch.ones_like(X1).reshape(-1)], dim=-1)

        Y = model1(points.unsqueeze(0), W=W1.unsqueeze(0)) if W1 is not None else model1(points)
        Y2 = model2(points.unsqueeze(0), W=W2.unsqueeze(0)) if W2 is not None else model2(points)

        z1 = Y.reshape(x_points, y_points).detach().cpu().numpy()
        z2 = Y2.reshape(x_points, y_points).detach().cpu().numpy()
        z_min = min(z_min, z1.min(), z2.min())
        z_max = max(z_max, z1.max(), z2.max())

        surfaces.append(go.Surface(x=X1, y= X2, z=z1, showscale=False, opacity=0.9))
        surfaces2.append(go.Surface(x=X1, y= X2, z=z2, showscale=False, opacity=0.9))

    # Add initial surfaces
    fig.add_trace(surfaces[0], row=1, col=1)
    fig.add_trace(surfaces2[0], row=1, col=2)

    # Animation frames
    frames = [go.Frame(data=[surfaces[k], surfaces2[k]]) for k in range(N)]
    fig.frames = frames

    # Animation controls
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, {
                    "frame": {"duration": 200, "redraw": True},
                    "fromcurrent": True,
                    "transition": {"duration": 0}
                }]
            )],
            x=0.5, y=1.15, xanchor="center", yanchor="top"
        )]
    )

    # 🔧 Fix scale for BOTH subplots
    scene_settings = dict(
        xaxis=dict(range=[x_min, x_max], autorange=False),
        yaxis=dict(range=[y_min, y_max], autorange=False),
        zaxis=dict(range=[z_min, z_max], autorange=False),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=0.5)
    )

    fig.update_layout(scene=scene_settings, scene2=scene_settings)

    fig.write_html(f"./plots/{model_name} - {ds_name}.html")

import config



def plot_predictions(gt_model, model):
    """
    Plot the predictions of a training model against the ground truth.
    """
    #model2 = Linear(dx=1 ,dy=1, order=2)
    #model2.W = torch.nn.Parameter(model.W)

    X = torch.linspace(0, 1, 128*model.dx).view(128, 1*model.dx)
    Y_gt = gt_model(X)
    Y = model(X, plot=True)
    plt.plot(X[:, 0].detach().numpy(), Y.detach().numpy())
    plt.plot(X[:, 0].detach().numpy(), Y_gt.detach().numpy())
    plt.show()
def plot_regression_on_dataset(y_test, y_pred, name):
    ex = np.arange(y_test.size(0))
    plt.xlim(-1, y_test.size(0))
    plt.ylim(torch.min(y_test.cpu()), torch.max(y_test.cpu()))
    idx = torch.argsort(y_test.cpu().flatten())
    # plt.scatter(ex, cf_pred.flatten().detach().numpy()[0:100])
    plt.scatter(ex, y_pred[idx].flatten().detach().cpu().numpy(), color='blue')
    plt.scatter(ex, y_test[idx].flatten().detach().cpu().numpy(), color='red')

    plt.savefig(name)
    plt.show()
    plt.close()


def plot_regression(ds, pred_dict, i, gt_name):
    k=0
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    model_type = "Polynomial"

    cf = "cf "+model_type+str(k)

    losses =  ['mle-dataset', 'backward-kl', 'forward-kl', 'mle-params']
    lossesShift =  ['MLE-Dataset', 'Rev-KL', 'Fwd-KL', 'MLE-Params']

    plot_regression_on_dataset_ax(ds.Y, pred_dict[cf].squeeze(0), axes[0], "Closed Form Solution")

    for l in range(len(losses)):
        plot_regression_on_dataset_ax(ds.Y, pred_dict[losses[l] + " " + model_type+str(k)].squeeze(0), axes[l+1], lossesShift[l])

    plt.savefig("./plots/"+gt_name+str(i)+".pdf")
    plt.show()
    plt.close()


def plot_regression_on_dataset_ax(y_test, y_pred, ax, title):
    ex = np.arange(y_test.size(0))
    ax.set_xlim(-1, y_test.size(0))
    ax.set_ylim(float(torch.min(y_test.cpu())), float(torch.max(y_test.cpu())))

    idx = torch.argsort(y_test.cpu().flatten())

    ax.scatter(ex, y_pred[idx].flatten().detach().cpu().numpy(), color='blue')
    ax.scatter(ex, y_test[idx].flatten().detach().cpu().numpy(), color='red')

    if title is not None:
        ax.set_title(f"{title} {metrics.mse(y_test, y_pred).item():.3f}")

def normalize_minus1_1(x):
    # Compute min and max
    x_min = x.min()
    x_max = x.max()

    # Avoid division by zero
    scale = x_max - x_min
    if scale == 0:
        return torch.zeros_like(x)

    # Normalize to [0,1]
    x_norm = (x - x_min) / scale

    # Rescale to [-1,1]
    x_norm = x_norm * 2 - 1
    return x_norm

import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def best_subplot_shape(n):
    """
    Given n subplots, returns (nrows, ncols)
    that form the most square/compact rectangular grid.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    # Start from the square root to get close to square
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols

def plot(gt_W, posterior_means, params_list, style_list, save_name):

    # Create 4x5 grid (total = 20 plots)
    n_params = params_list[0][0].size(-1)
    nrows, ncols = best_subplot_shape(n_params)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))

    # axes is a 2D array of Axes objects
    for i, ax in enumerate(axes.flat):
        x = np.linspace(-5, 5, 500)
        plot_prior=False
        if plot_prior:
            y = norm.pdf(x, 0, 1)
            ax.plot(x, y, lw=1, color="black", label="Prior", alpha=0.8)
        if posterior_means is not None:
            ax.axvline(posterior_means[i].detach().numpy(), color='black', label='Closed form solution', alpha = 0.9, lw=1)
        if i >= n_params: continue
        for params, style in zip(params_list, style_list):
            color, label, linestyle, alpha = style
            if isinstance(params, tuple):
                means, vars = params[0].detach().numpy(), params[1].detach().numpy()
                y = norm.pdf(x, means[0, i], np.sqrt(vars[0, i]))
                ax.plot(x, y, lw=1, color=color, label=label, linestyle=linestyle, alpha=alpha)

            else:
                ax.axvline(params[0, i].detach().numpy(), color=color, label=label, linestyle=linestyle, alpha=alpha, lw=1)


        if gt_W is not None:
            ax.axvline(gt_W[:, i].detach().numpy(), color='black', linestyle='dotted')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-0.5, 3)

        ax.set_title(f"Parameter {i+1}")

    handles, labels = ax.get_legend_handles_labels()  # from last axis
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_name)

def plot_params(models, style_list, eval_spec, x_dist, save_name, normalize):

    # create a ground truth target function and sample datasets
    gt_model = eval(eval_spec[0])(dx=dx, dy=dy, **eval_spec[1])
    bounds = datasets.gen_uniform_bounds(dx, x_dist=x_dist)
    ds_test = datasets.PointDataset(test_set_size, gt_model, x_dist=x_dist, noise_std=0., bounds=bounds)
    ds_input = datasets.PointDataset(input_set_size, gt_model, x_dist=x_dist, noise_std=0.5, bounds=bounds)

    params_list = []

    for icm in models:
        predictions, params, scales = icm.predict(
            torch.cat((ds_input.X, ds_input.Y), dim=-1).unsqueeze(0), ds_test.X.unsqueeze(0))
        params_list.append(params)
        print(save_name)
        print(metrics.mse(predictions, ds_test.Y))

    if normalize:
        xynorm, scales = in_context_models.normalize_input(torch.cat((ds_input.X, ds_input.Y), dim=-1).unsqueeze(0).transpose(0, 1))

        cf_params = models[0].eval_model.closed_form_solution_regularized(xynorm.squeeze(1)[:, 0:3], xynorm.squeeze(1)[:, 3],
                                                                         lambd=config.lambda_mle, scales=scales)

        y_norm_pred = models[0].eval_model.forward(in_context_models.normalize_to_scales(ds_test.X.unsqueeze(0), scales),
                                     cf_params.unsqueeze(0), scales=scales)

        print(metrics.mse(y_norm_pred, xynorm.squeeze(1)[:, 3]))

        cf_pred = in_context_models.renormalize(models[0].eval_model.forward(in_context_models.normalize_to_scales(ds_test.X.unsqueeze(0), scales), cf_params.unsqueeze(0), scales=scales), scales)
        print(metrics.mse(cf_pred, ds_test.Y))

        #model_pred = in_context_models.renormalize(rev_kl_model.eval_model.forward(in_context_models.normalize_to_scales(ds_test.X.unsqueeze(0), scales), params_mle_ds.unsqueeze(0), scales=scales), scales)
    else:
        cf_params = models[0].eval_model.closed_form_solution_regularized(ds_input.X, ds_input.Y, lambd=config.lambda_mle)
        cf_pred = models[0].eval_model.forward(ds_test.X.unsqueeze(0), cf_params.unsqueeze(0))

    plot(None, cf_params, params_list, style_list, save_name)


if __name__ == "__main__":
    pass
    dx,dy=3,1
    test_set_size=1000
    input_set_size = 128
    default_specs = {
        'transformer_arch':
         {
            'dT': 256,
            'num_heads': 4,
            'num_layers': 4,
            'output': 'attention-pool'
        },
        'train_specs':
        {
            'lr':0.0001,
            'min_lr': 1e-6,
            'weight_decay': 1e-5,
            'dataset_amount': 10000, #100000,
            'dataset_size': 1024,
            'num_iters': 1000000,
            'batch_size': 100,
            'valset_size': 1000, #10000,
            'normalize': True
        },
        'early_stopping_params': {
            'early_stopping_enabled': False,
            'patience': 10,
            'min_delta': 0.01,
            'load_best': True
        },
        'save_path': './exp2_default',
        'save_all': False
    }
    model_specs = [('Linear', {'order': 3, 'feature_sampling_enabled': True}),
                   ('Linear', {'order': 3, 'feature_sampling_enabled': True, 'nonlinear_features_enabled': True}),
                   ('Linear', {'order': 1, 'feature_sampling_enabled': True}), ]
    x_dist = 'uniform_fixed'
    normalize = False


    def plot_models(model_specs, style_list, model_assumption_spec, eval_spec, normalize, x_dist):
        model_spec_training = model_assumption_spec[1].copy()
        model_spec_training.pop('feature_sampling_enabled', None)
        model_list = []
        for elem in model_specs:
            loss, arch_spec, model_path = elem
            model = in_context_models.InContextModel(dx, dy, arch_spec, model_assumption_spec[0],
                                             loss=loss, normalize=normalize,
                                             **model_spec_training)
            checkpoint = torch.load(model_path + ".pt", map_location=config.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model_list.append(model)
        for i in range(30):
            plot_params(model_list, style_list, eval_spec, x_dist,  save_name="./plots/"+str(i)+".svg", normalize=normalize)


    style_list = [("blue", "", "solid", 0.), ("red", "", "solid", 0.), ("green", "Fwd-KL", "solid", 0.75), ("Orange", "MLE-Params", "solid", 0.9)]
    p1 = "./exp2_uniform_fixed_no_normalize_inc_ds_size/models/backward-kl Polynomial"
    p2 = "./exp2_uniform_fixed_no_normalize_inc_ds_size/models/mle-dataset Polynomial"
    p3 = "./exp2_uniform_fixed_no_normalize_inc_ds_size/models/forward-kl Polynomial"
    p4 = "./exp2_uniform_fixed_no_normalize_inc_ds_size/models/mle-params Polynomial"
    plot_models([("backward-kl", default_specs['transformer_arch'], p1), ("mle-dataset", default_specs['transformer_arch'], p2), ("forward-kl", default_specs['transformer_arch'], p3), ("mle-params", default_specs['transformer_arch'], p4)], style_list, model_specs[0], model_specs[1], normalize, x_dist)

    loss, arch_spec, model_path = 'forward-kl', default_specs[
        'transformer_arch'], "./exp2_uniform_fixed_fwd_stream18112025/models/forward-kl Polynomial"
    model_spec_training = model_specs[0][1].copy()
    model_spec_training.pop('feature_sampling_enabled', None)
    model = in_context_models.InContextModel(dx, dy, arch_spec, model_specs[0][0],
                                             loss=loss, normalize=normalize,
                                             **model_spec_training)
    checkpoint = torch.load(model_path + ".pt", map_location=config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model2 = Linear(dx=3, dy=1, order=3, feature_sampling_enabled=True, nonlinear_features_enabled=True)
    model3 = Linear(dx=3, dy=1, order=3, feature_sampling_enabled=False)
    bounds = datasets.gen_uniform_bounds(dx, x_dist=x_dist)
    ds_input = datasets.PointDataset(128, model2, x_dist=x_dist, noise_std=0.5, bounds=bounds)
    ds_test = datasets.PointDataset(1000, model2, x_dist=x_dist, noise_std=0., bounds=bounds)
    # model2 = train(model2, ds, valset=ds_val, valfreq=1000, iterations=num_iters, batch_size=100, lr=config.lr_classical, use_wandb=config.wandb_enabled)
    cf_params = model3.closed_form_solution_regularized(ds_input.X, ds_input.Y, lambd=config.lambda_mle)
    cf_prediction = model3.forward(ds_test.X.unsqueeze(0), cf_params.unsqueeze(0))
    _, pred_params, _1 = model.predict(torch.cat((ds_input.X, ds_input.Y), dim=-1).unsqueeze(0), ds_test.X.unsqueeze(0))

    #print(metrics.mse(_, ds_test.Y))
    W_cf = model3.closed_form_solution_regularized(ds_input.X, ds_input.Y, lambd=config.lambda_mle)
    W_cf = model3.closed_form_solution_regularized(ds_input.X, ds_input.Y, lambd=config.lambda_mle)
    plot_3d_surfaces(model2, model3, W1=None, W2=pred_params[0])

