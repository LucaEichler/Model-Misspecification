# Experiment for finding out the best regularizer for the MLE solution.
# Winner: 120

import torch

import config
import datasets
import main
from classical_models import Linear
from main import train, eval_plot

num_iters=10
tries = 1000   # How many different functions to test on
sizes_exp = [1e-4, 1e-3, 1e-2, 0.1, 1., 10., 100., 1000., 10000., 100000.]
sizes_inc = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800]
sizes_inc2 = [20, 40, 60, 80, 100, 120, 140, 160, 180]
sizes_inc3 = [100, 150, 200, 250, 300]
sizes_inc4 = [100, 110, 120, 130, 140, 150, 160, 170, 180]

test_set_size=1000
train_set_size=1000


values_to_test = sizes_inc3

mse=torch.zeros(len(values_to_test))
mse_params=torch.zeros(len(values_to_test))

dx=3
dy=1

# Model which will be used to calculate closed form MLE solutions
model = Linear(dx=dx, dy=dy, order=3, feature_sampling_enabled=False, nonlinear_features_enabled=True)

for j in range(tries):
    # Sample a random test function and test set
    gt_model = Linear(dx=dx, dy=dy, order=3, feature_sampling_enabled=True, nonlinear_features_enabled=True)
    test_set = datasets.PointDataset(size=test_set_size, model=gt_model, noise_std=0.5)
    train_set = datasets.PointDataset(size=train_set_size, model=gt_model, noise_std=0.5)

    i = 0
    for lambd in values_to_test:

        # estimate parameters via closed form mle, using train set as input
        params_mle = model.closed_form_solution_regularized(train_set.X, train_set.Y, lambd=lambd)

        # predict on test set using X as input
        Y_pred = model.forward(test_set.X.unsqueeze(0), params_mle.unsqueeze(0))

        # calculate mse between predictions and correct (noisy) data points
        mse[i] += torch.mean((Y_pred-test_set.Y)**2)

        #calculate mse between mle params and ground truth params
        mse_params[i] += torch.mean((params_mle-gt_model.get_W())**2)



        X = torch.linspace(-2, 2, 25).unsqueeze(1).to(config.device)
        X = torch.cat([X, X, X], dim=-1)

        Y_pred = model.forward(X.unsqueeze(0), params_mle.unsqueeze(0))
        #eval_plot("", "", gt_model, test_set.X[:, 0], Y_pred)
        #eval_plot("", "", gt_model, X[:, 0], Y_pred)

        i+=1

print(mse/tries)
print(mse_params/tries)