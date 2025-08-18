import config
from main import train_in_context_models

trained_in_context_models = train_in_context_models(dx=3, dy=1, dataset_amount=config.dataset_amount,
                            dataset_size=config.dataset_size_in_context, batch_size=config.batch_size_in_context,  num_iters=config.num_iters_in_context, noise_std=0.5, compute_closed_form_mle = True,
                            model_specs=[('Linear', {'order': 1, 'feature_sampling_enabled': True}), ('Linear', {'order': 3, 'feature_sampling_enabled': True}),('Linear', {'order': 3, 'feature_sampling_enabled': True, 'nonlinear_features_enabled': True})])

# 1. calculate mle solution and compare with it -> compute table or so (table maybe later)
# check if the mle solutions are correct!!!

# 2. experiments with heuristics for N

# 3. train models with trained params