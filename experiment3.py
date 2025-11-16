import math

import numpy as np
import pandas as pd
import torch
import scipy.stats as stats
from matplotlib import pyplot as plt

import config
import datasets
import main
import plotting
from classical_models import Linear, NonLinear
from main import train, eval_plot
import seed
from metrics import mse, mse_rel, mse_range

seed.set_seed(0)

num_iters = 1000000
tries = 50
sizes = [50, 200, 500, 2000, 5000, 20000, 50000]
test_set_size=10000
val_set_size=10000
dx=3
dy=1
dh=100
validation_frequency = 1000
x_dist = 'uniform_fixed'

mse_nn = torch.empty((tries, len(sizes)))
mse_closed_form = torch.empty((tries, len(sizes)))
mse_nn_rel = torch.empty((tries, len(sizes)))
mse_closed_form_rel = torch.empty((tries, len(sizes)))
mse_nn_range = torch.empty((tries, len(sizes)))
mse_closed_form_range = torch.empty((tries, len(sizes)))

for j in range(tries):
    gt_model = Linear(dx=dx, dy=dy, order=3, feature_sampling_enabled=True, nonlinear_features_enabled=True)
    bounds = datasets.gen_uniform_bounds(dx, x_dist)
    test_set = datasets.PointDataset(size=test_set_size, model=gt_model, x_dist=x_dist, noise_std=0.0, bounds=bounds)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(sizes),
        figsize=(4 * len(sizes), 4),
        squeeze=False
    )

    for i in range(len(sizes)):
        dataset_size = sizes[i] #16*2**i
        ds = datasets.PointDataset(size=dataset_size, model=gt_model, x_dist=x_dist, noise_std=0.5, bounds=bounds)
        ds_val = datasets.PointDataset(size=val_set_size, model=gt_model, x_dist=x_dist, noise_std=0.5, bounds=bounds)

        # 'dummy' model for computing closed form solution
        model = Linear(dx=dx, dy=dy, order=3, feature_sampling_enabled=False, nonlinear_features_enabled=True).to(config.device)

        params_mle = model.closed_form_solution_regularized(ds.X.to(config.device), ds.Y.to(config.device),
                                                            lambd=config.lambda_mle)

        Y_pred_closed_form = model.forward(test_set.X.unsqueeze(0), params_mle.unsqueeze(0))
        gt_Y = gt_model(test_set.X)

        # train neural network
        model_nn = NonLinear(dx=dx, dy=dy, dh=dh)
        model_nn = train(model_nn, ds, valset=ds_val, valfreq=validation_frequency, iterations=num_iters,
                         batch_size=100,
                         lr=config.lr_classical, weight_decay=config.weight_decay_classical, use_wandb=config.wandb_enabled, save_path=None)

        Y_pred_nn = model_nn(test_set.X)


        mse_nn[j, i] = mse(Y_pred_nn, gt_Y)
        mse_closed_form[j, i] = mse(Y_pred_closed_form, gt_Y)
        mse_nn_rel[j, i] = mse_rel(Y_pred_nn, gt_Y)
        mse_closed_form_rel[j, i] = mse_rel(Y_pred_closed_form, gt_Y)
        mse_nn_range[j, i] = mse_range(Y_pred_nn, gt_Y)
        mse_closed_form_range[j, i] = mse_range(Y_pred_closed_form, gt_Y)



        with open("./exp3_mse.csv", "a") as f:
            f.write(str(mse_nn[j, i].item()) + " " + str(mse_closed_form[j, i].item()) + "\n")

        #name="./plots/" + "nn " + str(j) + " " + str(i)
        #plotting.plot_regression_on_dataset(test_set.Y[1000:1100], Y_pred_nn[1000:1100], name)

        ax = axes[0][i]
        plotting.plot_regression_on_dataset_ax(test_set.Y[1000:1100], Y_pred_nn[1000:1100], ax)
        ax.set_title(f"N={dataset_size}")
        ax.set_xticks([])

    plt.tight_layout()
    plt.savefig(f"./plots/trial_{j}.pdf")
    plt.close()



def mean_and_ci(values, confidence=0.95):  #TODO: put in utility file and merge with main.py standard error computation
    """
    values: 1D tensor of shape [num_trials]
    returns mean, err (so you can plot mean ± err)
    """
    n = values.numel()
    mean = values.mean()
    std = values.std(unbiased=True)
    sem = std / math.sqrt(n)

    # t-distribution multiplier
    h = stats.t.ppf((1 + confidence) / 2., n - 1) * sem

    return mean.item(), h.item()

results = []

for i in range(len(sizes)):
    mean_nn, error_nn = mean_and_ci(mse_nn[:, i], confidence=0.95)
    mean_cf, error_cf = mean_and_ci(mse_closed_form[:, i], confidence=0.95)
    results.append({
        "dataset_size": sizes[i],
        "mse_nn": mean_nn,
        "mse_cf": mean_cf,
        "nn_err": error_nn,
        "cf_err": error_cf
    })

# Create DataFrame
df = pd.DataFrame(results)

df.to_csv("experiment3_results.csv", index=False)

results = []

for i in range(len(sizes)):
    mean_nn, error_nn = mean_and_ci(mse_nn_rel[:, i], confidence=0.95)
    mean_cf, error_cf = mean_and_ci(mse_closed_form_rel[:, i], confidence=0.95)
    results.append({
        "dataset_size": sizes[i],
        "mse_nn": mean_nn,
        "mse_cf": mean_cf,
        "nn_err": error_nn,
        "cf_err": error_cf
    })

# Create DataFrame
df = pd.DataFrame(results)

df.to_csv("experiment3_results_rel.csv", index=False)

results = []

for i in range(len(sizes)):
    mean_nn, error_nn = mean_and_ci(mse_nn_range[:, i], confidence=0.95)
    mean_cf, error_cf = mean_and_ci(mse_closed_form_range[:, i], confidence=0.95)
    results.append({
        "dataset_size": sizes[i],
        "mse_nn": mean_nn,
        "mse_cf": mean_cf,
        "nn_err": error_nn,
        "cf_err": error_cf
    })

# Create DataFrame
df = pd.DataFrame(results)

df.to_csv("experiment3_results_range.csv", index=False)

