import pandas as pd
import torch

import config
import datasets
import main
import plotting
from main import train_in_context_models

import classical_models
from classical_models import Linear

dx=3
dy=1

model_specs = [('Linear', {'order': 3, 'feature_sampling_enabled': True}), ('Linear', {'order': 3, 'feature_sampling_enabled': True, 'nonlinear_features_enabled': True}), ('Linear', {'order': 1, 'feature_sampling_enabled': True}),]
tries = config.test_trials
dataset_size = config.dataset_size_in_context

trained_in_context_models = train_in_context_models(dx=dx, dy=dy, x_dist='uniform', dataset_amount=config.dataset_amount,
                            dataset_size=config.dataset_size_in_context, batch_size=config.batch_size_in_context,  num_iters=config.num_iters_in_context, noise_std=0.5,
                            model_specs=model_specs)
results = []

for model_spec in model_specs:
    for i in range(tries):

        # create new ground truth model for evaluation
        gt_model = eval(model_spec[0])(dx=dx, dy=dy, **model_spec[1])
        bounds = datasets.gen_uniform_bounds(dx)

        # sample an input dataset from ground truth
        ds_input = datasets.PointDataset(dataset_size, gt_model, x_dist='uniform', noise_std=0.5, bounds=bounds)

        # test dataset, noise disabled to get target function values
        ds_test = datasets.PointDataset(dataset_size, gt_model, x_dist='uniform', noise_std=0., bounds=bounds)


        for loss, in_context_model in trained_in_context_models:
            # use the model that the in_context_model maps to for computing the mle solution
            closed_form_params = in_context_model.eval_model.closed_form_solution_regularized(ds_input.X, ds_input.Y, lambd=config.lambda_mle)
            closed_form_prediction = in_context_model.eval_model.forward(ds_test.X.unsqueeze(0), closed_form_params.unsqueeze(0))
            predictions, params = in_context_model.predict(torch.cat((ds_input.X, ds_input.Y), dim=-1).unsqueeze(0), ds_test.X.unsqueeze(0))

            #plotting.plot_3d_surfaces(in_context_model.eval_model, in_context_model.eval_model, closed_form_params, params, gt_model._get_name() + " " + str(i), loss + " " + in_context_model.eval_model._get_name())

            results.append({
                "trial": i,
                'gt': gt_model._get_name(),
                'model_name': loss+" "+in_context_model.eval_model._get_name(),
                "mse_params_closed_form_gradient_descent": torch.mean((closed_form_params-params)**2).item(),
                "mse_closed_form_gradient_descent": torch.mean((closed_form_prediction-predictions)**2).item(),
                "mse_closed_form": torch.mean((closed_form_prediction-ds_test.Y)**2).item(),
                "mse_gradient_descent": torch.mean((predictions-ds_test.Y)**2).item()
            })

            Xplot = torch.linspace(-2, 2, 25).unsqueeze(1).to(config.device)
            Xplot = torch.cat([Xplot, Xplot, Xplot], dim=-1)
            Yplot = gt_model(Xplot)

            Y_predplot, params_predplot = in_context_model.predict(torch.cat((Xplot, Yplot), dim=-1).unsqueeze(0), Xplot.unsqueeze(0))

            main.eval_plot(gt_model._get_name() + " " + str(i), loss + " " + in_context_model.eval_model._get_name(), gt_model, Xplot[:,0], Y_predplot.squeeze(0))



# Create DataFrame
df = pd.DataFrame(results)

df_avg = df.groupby(['gt', 'model_name']).mean().reset_index()

df_avg.to_csv("experiment2_results.csv", index=False)




# 1. calculate mle solution and compare with it -> compute table or so (table maybe later)
# check if the mle solutions are correct!!!

# 2. experiments with heuristics for N

# 3. train models with trained params