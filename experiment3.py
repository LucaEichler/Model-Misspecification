import math

import pandas as pd
import torch
import scipy.stats as stats

import config
import datasets
import main
from classical_models import Linear, NonLinear
from main import train, eval_plot
import seed

seed.set_seed(0)

num_iters=1000000
tries = 10
sizes = [1000, 5000, 20000, 50000]
test_set_size=10000
val_set_size=5000
dx=3
dy=1
dh=100

results = []

mse_nn = torch.empty((tries, len(sizes)))
mse_closed_form = torch.empty((tries, len(sizes)))

for j in range(tries):
    gt_model = Linear(dx=dx, dy=dy, order=3, feature_sampling_enabled=True, nonlinear_features_enabled=True)
    bounds = datasets.gen_uniform_bounds(dx)
    test_set = datasets.PointDataset(size=test_set_size, model=gt_model, x_dist='uniform', noise_std=0.5, bounds=bounds)
    for i in range(len(sizes)):
        dataset_size = sizes[i] #16*2**i
        ds = datasets.PointDataset(size=dataset_size, model=gt_model, x_dist='uniform', noise_std=0.5, bounds=bounds)
        ds_val = datasets.PointDataset(size=val_set_size, model=gt_model, x_dist='uniform', noise_std=0.5, bounds=bounds)

        # 'dummy' model for computing closed form solution
        model = Linear(dx=dx, dy=dy, order=3, feature_sampling_enabled=False, nonlinear_features_enabled=True).to(config.device)

        # train neural network
        model_nn = NonLinear(dx=dx, dy=dy, dh=dh)
        model_nn = train(model_nn, ds, valset=ds_val, valfreq=1000, iterations=num_iters, batch_size=100, lr=config.lr_classical, use_wandb=config.wandb_enabled)

        params_mle = model.closed_form_solution_regularized(ds.X.to(config.device), ds.Y.to(config.device), lambd=config.lambda_mle*dataset_size)

        Y_pred_closed_form = model.forward(test_set.X.unsqueeze(0), params_mle.unsqueeze(0))
        gt_Y = gt_model(test_set.X)

        Y_pred_nn = model_nn(test_set.X)

        mse_nn[j, i] = torch.mean((Y_pred_nn-gt_Y)**2)
        mse_closed_form[j, i] = torch.mean((Y_pred_closed_form-gt_Y)**2)

        print(torch.mean((Y_pred_nn-gt_Y)**2))
        print(torch.mean((Y_pred_closed_form-gt_Y)**2))
        # visualize data through one slice
        #X = torch.linspace(-2, 2, 25).unsqueeze(1).to(config.device)
        #X = torch.cat([X, X, X], dim=-1)

        #Y_pred = model_trained(X)
        #Y_pred_mle = model.forward(X.unsqueeze(0), params_mle.unsqueeze(0))
        #eval_plot("", "", gt_model, X[:, 0], Y_pred)
        #eval_plot("", "", gt_model, X[:, 0], Y_pred_mle)

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

