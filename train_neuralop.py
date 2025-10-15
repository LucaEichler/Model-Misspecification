import os

import pandas as pd

import config
import datasets
from main import train
from metrics import mse
from neuralop import InterpolationModel
from classical_models import Linear, NonLinear

model = InterpolationModel()
print(model)
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
            'lr': 0.0001,
            'min_lr': 1e-6,
            'weight_decay': 1e-5,
            'dataset_amount': 10000,
            'dataset_size': 128,
            'num_iters': 10000,
            'batch_size': 100,
            'valset_size': 10000,
            'normalize': True
        },
    'early_stopping_params': {
        'early_stopping_enabled': False,
        'patience': 10,
        'min_delta': 0.01,
        'load_best': True
    },
    #'losses': losses,
    'save_path': './neural_op_default',
    'save_all': False
}
specs = default_specs
train_specs = specs['train_specs']

model_specs = [('Linear', {'order': 3, 'feature_sampling_enabled': True}),
               ('Linear', {'order': 3, 'feature_sampling_enabled': True, 'nonlinear_features_enabled': True}),
               ('Linear', {'order': 1, 'feature_sampling_enabled': True}), ]

model_spec = model_specs[0]
x_dist='gaussian'
noise_std=0.5
dx, dy = 3,1
model_path = specs['save_path']
model_name = 'Neural Operator'
save_all =specs['save_all']
os.makedirs(model_path, exist_ok=True)
dataset = datasets.ContextDataset(train_specs['dataset_amount'], train_specs['dataset_size'], model_spec[0], dx, dy,
                                  x_dist, noise_std, **model_spec[1])
valset = datasets.ContextDataset(train_specs['valset_size'], train_specs['dataset_size'], model_spec[0], dx, dy,
                                 x_dist, noise_std, **model_spec[1])  # TODO valset size in config

model_trained = train(model, dataset, valfreq=500, valset=valset, iterations=train_specs['num_iters'],
                      batch_size=train_specs['batch_size'],
                      lr=train_specs['lr'], weight_decay=train_specs['weight_decay'],
                      early_stopping_params=specs['early_stopping_params'], use_wandb=config.wandb_enabled,
                      min_lr=train_specs['min_lr'], save_path=model_path, wandb_name=model_name, save_all=save_all)
 # test neural operator - plots ?

results=[]
trials = 10
input_set_size = 128
test_set_size = 1000
for model_spec in model_specs:
    for i in range(trials):
        # create new ground truth model for evaluation
        gt_model = eval(model_spec[0])(dx=dx, dy=dy, **model_spec[1])
        bounds = datasets.gen_uniform_bounds(dx)

        # sample an input dataset from ground truth
        ds_input = datasets.PointDataset(input_set_size, gt_model, x_dist='uniform', noise_std=0.5, bounds=bounds)

        # test dataset, noise disabled to get target function values
        ds_test = datasets.PointDataset(test_set_size, gt_model, x_dist='uniform', noise_std=0., bounds=bounds)

        predictions = model_trained(ds_test.X.unsqueeze(0), ds_input.X.unsqueeze(0), ds_input.Y.unsqueeze(0), mask=None)

        results.append({
        "trial": i,
        'gt': gt_model._get_name(),
        'model_name': 'neuralop',
        "mse": mse(predictions.squeeze(0), ds_test.Y).item()}
        )

df = pd.DataFrame(results)
df_avg = df.groupby(['gt', 'model_name']).mean().reset_index()
df_avg.to_csv(specs['save_path'] + "/neuralop.csv", index=False)
