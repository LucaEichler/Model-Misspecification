# Test the parameter normalization:
# Generate ground truth model
# Generate input data with no noise
# Normalize input data
# Normalize parameters
# predict normalized y using normalized x and normalized params
import torch
import in_context_models
from classical_models import Linear
from datasets import PointDataset, gen_uniform_bounds

gt_model = Linear(dx=3, dy=1, order=3, feature_sampling_enabled=False, nonlinear_features_enabled=True)
dataset = PointDataset(1000, gt_model, x_dist='uniform', noise_std=0., bounds=gen_uniform_bounds(dim=3))
X, Y = dataset.X, dataset.Y
params = gt_model.get_W()

data_norm, scales = in_context_models.normalize_input(torch.cat((X,Y),dim=-1).unsqueeze(1))
params_norm = in_context_models.normalize_params(params.unsqueeze(0), scales)
x_norm = data_norm[:, :, 0:3]
y_norm = data_norm[:, :, 3]

y_pred = gt_model.forward(x_norm.transpose(0,1), W=params_norm, scales=scales)

error = y_norm-y_pred.squeeze(0)

assert torch.allclose(error, torch.zeros_like(error), atol=1e-6)
