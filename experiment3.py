# In this experiment, we want to find out
# how many data points the model needs to reach optimal performance
import pandas as pd
import torch

import config
import datasets
import main
from classical_models import Linear, NonLinear
from main import train, eval_plot
import seed

seed.set_seed(0)

num_iters=100000
tries = 1
sizes = [1000, 5000, 20000, 50000]
test_set_size=10000
results = torch.zeros(sizes)
dx=3
dy=1

results = []

mse_nn = torch.empty((tries, 4))
mse_closed_form = torch.empty((tries, 4))

for j in range(tries):
    gt_model = Linear(dx=dx, dy=dy, order=3, feature_sampling_enabled=True, nonlinear_features_enabled=True)
    test_set = datasets.PointDataset(size=test_set_size, model=gt_model, noise_std=0.5)
    for i in range(len(sizes)):
        dataset_size = sizes[i] #16*2**i
        ds = datasets.PointDataset(size=dataset_size, model=gt_model, noise_std=0.5)
        ds_val = datasets.PointDataset(size=dataset_size, model=gt_model, noise_std=0.5)

        model = Linear(dx=dx, dy=dy, order=3, feature_sampling_enabled=False, nonlinear_features_enabled=True).to(config.device)
        model_nn = NonLinear(dx=dx, dy=dy, dh=100)
        model_nn = train(model_nn, ds, valset=ds_val, valfreq=1000, iterations=num_iters, batch_size=100, lr=config.lr_classical, use_wandb=config.wandb_enabled)

        params_mle = model.closed_form_solution_regularized(ds.X.to(config.device), ds.Y.to(config.device), lambd=config.lambda_mle*dataset_size)

        Y_pred_closed_form = model.forward(test_set.X.unsqueeze(0), params_mle.unsqueeze(0))
        gt_Y = gt_model(test_set.X)

        Y_pred_nn = model_nn(test_set.X)

        mse_nn[j, i] = torch.mean((Y_pred_nn-gt_Y)**2)
        mse_closed_form[j, i] = torch.mean((Y_pred_closed_form-gt_Y)**2)


        # visualize data through one slice
        #X = torch.linspace(-2, 2, 25).unsqueeze(1).to(config.device)
        #X = torch.cat([X, X, X], dim=-1)

        #Y_pred = model_trained(X)
        #Y_pred_mle = model.forward(X.unsqueeze(0), params_mle.unsqueeze(0))
        #eval_plot("", "", gt_model, X[:, 0], Y_pred)
        #eval_plot("", "", gt_model, X[:, 0], Y_pred_mle)

for i in range(len(sizes)):
    results.append({
        "dataset_size": sizes[i],
        "mse_grad_descent": torch.mean(mse_nn, dim=0)[i].item(),
        "mse_closed_form": torch.mean(mse_closed_form, dim=0)[i].item()
    })

# Create DataFrame
df = pd.DataFrame(results)

df.to_csv("experiment3_results.csv", index=False)

