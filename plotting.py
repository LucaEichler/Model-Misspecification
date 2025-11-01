import matplotlib.pyplot as plt
import torch
from plotly.subplots import make_subplots
from scipy.stats import norm

import config
import datasets
import in_context_models
import metrics
from classical_models import LinearVariational, Linear
import plotly.graph_objects as go

from main import train



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


def plot(gt_W, cf_W, rev_kl, mle_ds):
    rev_kl_means, rev_kl_vars = rev_kl
    rev_kl_means = rev_kl_means.detach().numpy()
    rev_kl_vars = rev_kl_vars.detach().numpy()

    # Create 4x5 grid (total = 20 plots)
    fig, axes = plt.subplots(4,5, figsize=(15, 10))

    min = np.min(rev_kl_means)
    max = np.max(rev_kl_means)

    min = min-(min*0.2)
    max = max+(max*0.2)

    # axes is a 2D array of Axes objects
    for i, ax in enumerate(axes.flat):
        x = np.linspace(rev_kl_means[0, i]-3*rev_kl_vars[0, i], rev_kl_means[0, i]+3*rev_kl_vars[0, i], 500)
        x = np.linspace(-5, 5, 500)
        y_rev_kl = norm.pdf(x, rev_kl_means[0, i], np.sqrt(rev_kl_vars[0, i]))

        ax.plot(x, y_rev_kl, 'r-', lw=2, label='l', color='gray')
        ax.axvline(cf_W[:, i].detach().numpy(), color='red', linestyle='--')
        ax.axvline(gt_W[:, i].detach().numpy(), color='black', linestyle='dotted')
        ax.axvline(mle_ds[0, i].detach().numpy(), color='orange', linestyle='dashdot')
        ax.set_xlim(-4, 4)

        ax.set_title(f"Plot {i+1}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
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
            'dataset_size': 128,
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

    # create and load trained in context model
    model_spec = model_specs[0]
    model_spec_training = model_spec[1].copy()
    model_spec_training.pop('feature_sampling_enabled', None)
    rev_kl_model = in_context_models.InContextModel(dx, dy, default_specs['transformer_arch'], model_spec[0], loss="backward-kl", normalize=True,
                                             **model_spec_training)
    model_path = "./exp2_off_run_20102025/models/backward-kl Polynomial"
    checkpoint = torch.load(model_path+".pt", map_location=config.device)
    rev_kl_model.load_state_dict(checkpoint["model_state_dict"])

    mle_ds_model = in_context_models.InContextModel(dx, dy, default_specs['transformer_arch'], model_spec[0], loss="mle-dataset", normalize=False,
                                             **model_spec_training)
    model_path = "./exp2_uniform_fixed_no_normalize_01112025/models/mle-dataset Polynomial"
    checkpoint = torch.load(model_path+".pt", map_location=config.device)
    mle_ds_model.load_state_dict(checkpoint["model_state_dict"])

    x_dist = 'uniform_fixed'

    # create a ground truth target function and sample datasets
    eval_spec=model_specs[1]
    gt_model = eval(eval_spec[0])(dx=dx, dy=dy, **eval_spec[1])
    bounds = datasets.gen_uniform_bounds(dx, x_dist=x_dist)
    ds_test = datasets.PointDataset(test_set_size, gt_model, x_dist=x_dist, noise_std=0., bounds=bounds)
    ds_input = datasets.PointDataset(input_set_size, gt_model, x_dist=x_dist, noise_std=0.5, bounds=bounds)

    # closed form predictions
    cf_params = rev_kl_model.eval_model.closed_form_solution_regularized(ds_input.X, ds_input.Y, lambd=config.lambda_mle)
    cf_pred = rev_kl_model.eval_model.forward(ds_test.X.unsqueeze(0), cf_params.unsqueeze(0))
    print(metrics.mse(cf_pred, ds_test.Y))

    # in context model predictions
    predictions, params_rev_kl, scales = rev_kl_model.predict(torch.cat((ds_input.X, ds_input.Y), dim=-1).unsqueeze(0), ds_test.X.unsqueeze(0))
    print(metrics.mse(predictions, ds_test.Y))
    predictions, params_mle_ds, scales = mle_ds_model.predict(torch.cat((ds_input.X, ds_input.Y), dim=-1).unsqueeze(0), ds_test.X.unsqueeze(0))
    print(metrics.mse(predictions, ds_test.Y))

    normalize=False
    if normalize:
        plot(in_context_models.normalize_params(gt_model.get_W()[None, :], scales), in_context_models.normalize_params(cf_params.T, scales), params_rev_kl, params_mle_ds)
    else:
        plot(gt_model.get_W()[None, :], cf_params.T, params_rev_kl, params_mle_ds)



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

#model2 = Linear(dx=3, dy=1, order=3, nonlinear_features_enabled=True, feature_sampling_enabled=False)
#model2 = train(model2, ds, valset=ds_val, valfreq=1000, iterations=num_iters, batch_size=100, lr=config.lr_classical, use_wandb=config.wandb_enabled)

#plot_3d_surfaces(model2, model2, None, W2=model2.closed_form_solution_regularized(ds.X, ds.Y, lambd=config.lambda_mle))

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