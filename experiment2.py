import copy
import os

import pandas as pd
import torch

import config
import datasets
import main
import plotting
import seed
from main import train_in_context_models

import classical_models
from classical_models import Linear
from metrics import mse, mse_range, mse_rel
seed.set_seed(0)

dx = 3
dy = 1
plot=True

model_specs = [('Linear', {'order': 3, 'feature_sampling_enabled': True}),
               ('Linear', {'order': 3, 'feature_sampling_enabled': True, 'nonlinear_features_enabled': True}),
               ('Linear', {'order': 1, 'feature_sampling_enabled': True}), ]


input_set_size = 128    # the size of the context set that a model gets for testing
test_set_size = 1000    # the amount of points for each dataset that is tested on
trials = 100           # amount of ground truth functions that the model is tested on


def run_experiments(exp2_specs):
    for specification in exp2_specs:
        os.makedirs(specification['save_path'] + "/models/", exist_ok=True)
        trained_in_context_models = train_in_context_models(dx=dx, dy=dy, transformer_arch=specification['transformer_arch'],
                                                            x_dist='uniform', train_specs=specification['train_specs'], noise_std=0.5,
                                                            model_specs=model_specs, losses=specification['losses'], early_stopping_params=specification['early_stopping_params'], save_path=specification['save_path'])
        specification['trained_models'] = trained_in_context_models
        specification['results'] = [[], [], []]


    for model_spec in model_specs:
        for i in range(trials):

            # create new ground truth model for evaluation
            gt_model = eval(model_spec[0])(dx=dx, dy=dy, **model_spec[1])
            bounds = datasets.gen_uniform_bounds(dx)

            # sample an input dataset from ground truth
            ds_input = datasets.PointDataset(input_set_size, gt_model, x_dist='uniform', noise_std=0.5, bounds=bounds)

            # test dataset, noise disabled to get target function values
            ds_test = datasets.PointDataset(test_set_size, gt_model, x_dist='uniform', noise_std=0., bounds=bounds)

            for specification in exp2_specs:
                save_path = specification['save_path']
                trained_in_context_models = specification['trained_models']

                results = []
                results_range_normalized = []
                results_rel = []

                for loss, in_context_model in trained_in_context_models:
                    # use the model that the in_context_model maps to for computing the mle solution
                    closed_form_params = in_context_model.eval_model.closed_form_solution_regularized(ds_input.X, ds_input.Y,
                                                                                                      lambd=config.lambda_mle)
                    closed_form_prediction = in_context_model.eval_model.forward(ds_test.X.unsqueeze(0),
                                                                                 closed_form_params.unsqueeze(0))
                    predictions, params = in_context_model.predict(torch.cat((ds_input.X, ds_input.Y), dim=-1).unsqueeze(0),
                                                                   ds_test.X.unsqueeze(0))

                    # plotting.plot_3d_surfaces(in_context_model.eval_model, in_context_model.eval_model, closed_form_params, params, gt_model._get_name() + " " + str(i), loss + " " + in_context_model.eval_model._get_name())

                    specification['results'][0].append({
                        "trial": i,
                        'gt': gt_model._get_name(),
                        'model_name': loss + " " + in_context_model.eval_model._get_name(),
                        "mse_range_cf": mse_range(closed_form_prediction, ds_test.Y).item(),
                        "mse_range_gd": mse_range(predictions, ds_test.Y).item(),
                    })

                    specification['results'][1].append({
                        "trial": i,
                        'gt': gt_model._get_name(),
                        'model_name': loss + " " + in_context_model.eval_model._get_name(),
                        "mse_rel_cf": mse_rel(closed_form_prediction, ds_test.Y).item(),
                        "mse_rel_gd": mse_rel(predictions, ds_test.Y).item(),
                    })

                    specification['results'][2].append({
                        "trial": i,
                        'gt': gt_model._get_name(),
                        'model_name': loss + " " + in_context_model.eval_model._get_name(),
                        "mse_params_closed_form_gradient_descent": mse(closed_form_params, params).item(),
                        "mse_closed_form_gradient_descent": mse(closed_form_prediction, predictions).item(),
                        "mse_closed_form": mse(closed_form_prediction, ds_test.Y).item(),
                        "mse_gradient_descent": mse(predictions, ds_test.Y).item()
                    })


                    # TODO generate input dataset of 128 points randomly from input space

                    if plot:
                        os.makedirs(save_path + "/plots/", exist_ok=True)

                        Xplot = torch.linspace(-2, 2, 25).unsqueeze(1).to(config.device)
                        Xplot = torch.cat([Xplot, Xplot, Xplot], dim=-1)
                        Yplot = gt_model(Xplot)

                        # plotting is flawed, need to give different input to amortized model / cf
                        # sample an input dataset from ground truth
                        ds_input_plot = datasets.PointDataset(input_set_size, gt_model, x_dist='uniform', noise_std=0.5, bounds=torch.tensor([[-2., 2.], [-2., 2.], [-2., 2.]]))

                        Y_predplot, params_predplot = in_context_model.predict(torch.cat((ds_input_plot.X, ds_input_plot.Y), dim=-1).unsqueeze(0),
                                                                               Xplot.unsqueeze(0))

                        closed_form_params = in_context_model.eval_model.closed_form_solution_regularized(ds_input_plot.X, ds_input_plot.Y,
                                                                                                          lambd=config.lambda_mle)
                        Ypred_cf = in_context_model.eval_model.forward(Xplot.unsqueeze(0), closed_form_params.unsqueeze(0))

                        main.eval_plot(gt_model._get_name() + " " + str(i), loss + " " + in_context_model.eval_model._get_name(),
                                       gt_model, Xplot[:, 0], Y_predplot.squeeze(0), Ypred_cf, savepath=save_path)

    for specification in exp2_specs:
        save_path = specification['save_path']
        [results_range_normalized, results_rel, results] = specification['results']
        df = pd.DataFrame(results)
        df_avg = df.groupby(['gt', 'model_name']).mean().reset_index()
        df_avg.to_csv(save_path+"/experiment2_results.csv", index=False)

        df = pd.DataFrame(results_range_normalized)
        df_avg = df.groupby(['gt', 'model_name']).mean().reset_index()
        df_avg.to_csv(save_path+"/experiment2_results_range.csv", index=False)

        df = pd.DataFrame(results_rel)
        df_avg = df.groupby(['gt', 'model_name']).mean().reset_index()
        df_avg.to_csv(save_path+"/experiment2_results_rel.csv", index=False)


losses = ['mle-params', 'mle-dataset', 'forward-kl', 'backward-kl']


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
        'lr':0.0001,
        'weight_decay': 1e-5,
        'dataset_amount': 100000,
        'dataset_size': 128,
        'num_iters': 100000,
        'batch_size': 100
    },
    'early_stopping_params': {
        'early_stopping_enabled': True,
        'patience': 10,
        'min_delta': 0.01,
        'load_best': False
    },
    'losses': losses,
    'save_path': './exp2_default'
}

specs_1 = copy.deepcopy(default_specs)
specs_1['train_specs']['lr'] = 0.001


run_experiments([specs_1])
