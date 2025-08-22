# In this experiment, we want to find out
# how many data points the model needs to reach optimal performance
import pandas as pd
import torch

import config
import datasets
import main
from classical_models import Linear
from main import train, eval_plot

num_iters=50000
tries = 50
sizes = 10  # [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
test_set_size=10000
results = torch.zeros(sizes)
dx=3
dy=1

results = []

for j in range(tries):
    gt_model = Linear(dx=dx, dy=dy, order=3, feature_sampling_enabled=True, nonlinear_features_enabled=True)
    test_set = datasets.PointDataset(size=test_set_size, model=gt_model, noise_std=0.5)
    for i in range(sizes):
        dataset_size = 16*2**i
        ds = datasets.PointDataset(size=dataset_size, model=gt_model, noise_std=0.5)
        ds_val = datasets.PointDataset(size=dataset_size, model=gt_model, noise_std=0.5)

        model = Linear(dx=dx, dy=dy, order=3, feature_sampling_enabled=False, nonlinear_features_enabled=True)
        model_trained = train(model, ds, valset=ds_val, valfreq=1000, iterations=num_iters, batch_size=100, lr=config.lr_classical, use_wandb=config.wandb_enabled)

        params_grad_descent = model_trained.get_W()
        params_mle = model.closed_form_solution_regularized(ds.X.to(config.device), ds.Y.to(config.device), lambd=config.lambda_mle*dataset_size)

        Y_pred = model_trained(test_set.X)
        Y_pred_mle = model_trained.forward(test_set.X.unsqueeze(0), params_mle.unsqueeze(0))
        gt_Y = gt_model(test_set.X)

        mse_params_closed_form_gradient_descent = torch.mean((params_grad_descent-params_mle)**2)
        mse_closed_form_gradient_descent = torch.mean((Y_pred-Y_pred_mle)**2)
        mse_grad_descent = torch.mean((Y_pred-gt_Y)**2)
        mse_closed_form = torch.mean((Y_pred_mle-gt_Y)**2)


        # visualize data through one slice
        #X = torch.linspace(-2, 2, 25).unsqueeze(1).to(config.device)
        #X = torch.cat([X, X, X], dim=-1)

        #Y_pred = model_trained(X)
        #Y_pred_mle = model.forward(X.unsqueeze(0), params_mle.unsqueeze(0))
        #eval_plot("", "", gt_model, X[:, 0], Y_pred)
        #eval_plot("", "", gt_model, X[:, 0], Y_pred_mle)

        results.append({
            "trial": j,
            "dataset_size": dataset_size,
            "mse_params_closed_form_gradient_descent": mse_params_closed_form_gradient_descent.item(),
            "mse_closed_form_gradient_descent": mse_closed_form_gradient_descent.item(),
            "mse_grad_descent": mse_grad_descent.item(),
            "mse_closed_form": mse_closed_form.item()
        })

# Create DataFrame
df = pd.DataFrame(results)

# Average across tries
df_mean = df.groupby("dataset_size").mean().reset_index()

df_mean.to_csv("experiment3_results.csv", index=False)

