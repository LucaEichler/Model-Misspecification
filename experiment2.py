import torch

import config
import datasets
from main import train_in_context_models

model_specs = [('Linear', {'order': 1, 'feature_sampling_enabled': True}), ('Linear', {'order': 3, 'feature_sampling_enabled': True}),('Linear', {'order': 3, 'feature_sampling_enabled': True, 'nonlinear_features_enabled': True})]
tries = config.test_tries
dataset_size = config.dataset_size_classical

trained_in_context_models = train_in_context_models(dx=3, dy=1, dataset_amount=config.dataset_amount,
                            dataset_size=config.dataset_size_in_context, batch_size=config.batch_size_in_context,  num_iters=config.num_iters_in_context, noise_std=0.5, compute_closed_form_mle = True,
                            model_specs=model_specs)

for model_spec in model_specs:
    for i in range(tries):

        # create new ground truth model for evaluation
        gt_model = eval(model_spec[0])(dx=3, dy=1, **model_spec[1])

        # sample a test set from ground truth
        ds = datasets.PointDataset(dataset_size, gt_model)

        X, Y = ds.X, ds.Y

        for in_context_model in trained_in_context_models:
            # use the model that the in_context_model maps to for computing the mle solution
            mle_solution = in_context_model.eval_model.closed_form_solution_regularized(X, Y, lambd=config.weight_decay_classical)
            mle_prediction = in_context_model.eval_model.forward(X, mle_solution)

            predictions, params = in_context_model.predict(torch.cat((X, Y), dim=-1).unsqueeze(0), X.unsqueeze(0))


# 1. calculate mle solution and compare with it -> compute table or so (table maybe later)
# check if the mle solutions are correct!!!

# 2. experiments with heuristics for N

# 3. train models with trained params