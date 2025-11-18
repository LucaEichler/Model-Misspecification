# Test the parameter normalization:
# Generate ground truth model
# Generate input data with no noise
# Normalize input data
# Normalize parameters
# predict normalized y using normalized x and normalized params
from torch.utils.data import DataLoader

import datasets
import torch
import in_context_models
from classical_models import Linear
from datasets import PointDataset, gen_uniform_bounds

x_dist = 'uniform_fixed'

# create a target function (non-sparse) and sample a dataset from it
gt_model = Linear(dx=3, dy=1, order=3, feature_sampling_enabled=False, nonlinear_features_enabled=True)
dataset = PointDataset(1000, gt_model, x_dist=x_dist, noise_std=0., bounds=gen_uniform_bounds(x_dist=x_dist, dim=3))
X, Y = dataset.X, dataset.Y
params = gt_model.get_W()

data_norm, scales = in_context_models.normalize_input(torch.cat((X,Y),dim=-1).unsqueeze(1))
params_norm = in_context_models.normalize_params(params.unsqueeze(0), scales)
x_norm = data_norm[:, :, 0:3]
y_norm = data_norm[:, :, 3]

y_pred = gt_model.forward(x_norm.transpose(0,1), W=params_norm, scales=scales)

error = y_norm-y_pred.squeeze(0)

assert torch.allclose(error, torch.zeros_like(error), atol=1e-6)


model_spec = ('Linear', {'order': 3, 'feature_sampling_enabled': False, 'nonlinear_features_enabled': True})
gt_model = Linear(dx=3, dy=1, **model_spec[1])
dataset = datasets.ContextDataset(100, 128, model_spec[0], dx=3, dy=1,
                                      x_dist=x_dist, noise_std=0.5, **model_spec[1])
dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
data_iter = iter(dataloader)
batch = next(data_iter)

datasets_in, gt_params, gt_Y=batch
datasets_in, scales = in_context_models.normalize_input(datasets_in.transpose(0, 1))
datasets_in = datasets_in.transpose(0, 1)
datasets_in_X = datasets_in[:, :, 0:3]

params_norm = in_context_models.normalize_params(gt_params, scales)

y_norm = gt_model.forward(datasets_in_X, params_norm, scales)
y_pred = in_context_models.renormalize(y_norm, scales)


error = gt_Y-y_pred.squeeze(0)

mask = ~torch.isclose(error, torch.zeros_like(error), atol=1e-4)
bad_values = error[mask]
assert torch.allclose(error.flatten(), torch.zeros_like(error.flatten()), atol=1e-4)

