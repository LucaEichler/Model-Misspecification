import copy
import os

import pandas as pd
import torch

import config
import datasets
import main
import metrics
import plotting
import seed
import train_neuralop
from main import train_in_context_models

import classical_models
from classical_models import Linear
from metrics import mse, mse_range, mse_rel
from neuralop import InterpolationModel

seed.set_seed(0)

dx = 3
dy = 1
plot=False

model_specs = [('Linear', {'order': 3, 'feature_sampling_enabled': False}),]

eval_specs = [('Linear', {'order': 3, 'feature_sampling_enabled': True}),
               ('Linear', {'order': 3, 'feature_sampling_enabled': True, 'nonlinear_features_enabled': True}),
               ('Linear', {'order': 1, 'feature_sampling_enabled': True}), ]
losses = ['mle-params']

test_set_size = 1000    # the amount of points for each dataset that is tested on
trials = 50        # amount of ground truth functions that the model is tested on


def run_experiments(exp2_specs, nop_specs=None, x_dist=None):
    for specification in exp2_specs:

        dataset_sizes = []  # list of all unique dataset sizes, used later for efficiency
        if not dataset_sizes.__contains__(specification['train_specs']['dataset_size']):
            dataset_sizes.append(specification['train_specs']['dataset_size'])

        trained_in_context_models = train_in_context_models(dx=dx, dy=dy, transformer_arch=specification['transformer_arch'],
                                                            x_dist=x_dist, train_specs=specification['train_specs'], noise_std=0.5,
                                                            model_specs=model_specs, losses=specification['losses'], early_stopping_params=specification['early_stopping_params'], save_path=specification['save_path'], save_all=specification['save_all'])
        specification['trained_models'] = trained_in_context_models
        specification['results'] = [[], [], []]

    dataset_sizes = sorted(dataset_sizes, reverse=True) # sort dataset sizes descending

    if nop_specs: # load neural operator models
        nop_models = []
        nop_results = []
        for model_spec in model_specs:
            model = InterpolationModel()
            model_name = "nop_" + eval(model_spec[0])(dx=dx, dy=dy, **model_spec[1])._get_name()
            model_path = nop_specs['save_path'] + "/" + model_name
            if os.path.exists(
                    model_path + ".pt"):  # this path only exists when the train loop for a model was fully finished
                checkpoint = torch.load(model_path + ".pt",
                                        map_location=config.device)  # in this case, we load the model and skip training
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(config.device)

            nop_models.append((model, model_name, model_path))

    for eval_spec in eval_specs: # evaluate on every ground truth data distribution

        for i in range(trials): # average over T trials

            # create new ground truth model for evaluation
            gt_model = eval(eval_spec[0])(dx=dx, dy=dy, **eval_spec[1])
            bounds = datasets.gen_uniform_bounds(dx, x_dist=x_dist)

            # test dataset, noise disabled to get target function values
            ds_test = datasets.PointDataset(test_set_size, gt_model, x_dist=x_dist, noise_std=0.5, bounds=bounds)

            if nop_specs:
                for model, model_name, model_path in nop_models:
                    # sample an input dataset from ground truth
                    ds_input = datasets.PointDataset(128, gt_model, x_dist=x_dist, noise_std=0.5,
                                                     bounds=bounds)
                    predictions = model(ds_test.X.unsqueeze(0), ds_input.X.unsqueeze(0),
                                                ds_input.Y.unsqueeze(0), mask=None)

                    nop_results.append({
                        "trial": i,
                        'gt': gt_model._get_name(),
                        'model_name': model_name,
                        "mse": mse(predictions.squeeze(0), ds_test.Y).item()}
                    )
                    if plot:
                        os.makedirs(nop_specs['save_path'] + "/plots/", exist_ok=True)

                        Xplot = torch.linspace(-2, 2, 25).unsqueeze(1).to(config.device)
                        Xplot = torch.cat([Xplot, Xplot, Xplot], dim=-1)
                        Yplot = gt_model(Xplot)

                        ds_input_plot = datasets.PointDataset(128, gt_model, x_dist=x_dist, noise_std=0.5,
                                                              bounds=torch.tensor([[-2., 2.], [-2., 2.], [-2., 2.]]))

                        y_pred = model(Xplot.unsqueeze(0), ds_input_plot.X.unsqueeze(0),
                                               ds_input_plot.Y.unsqueeze(0), mask=None)

                        main.eval_plot(gt_model._get_name() + " " + str(i), model_name,
                                       gt_model, Xplot[:, 0], y_pred.squeeze(0), None, savepath=nop_specs['save_path'])

            ds_input_dict = {}
            for j in range(len(dataset_sizes)):
                ds_size = dataset_sizes[j]
                if j == 0:
                    # sample input dataset for the biggest dataset size
                    ds = datasets.PointDataset(ds_size, gt_model,
                                                     x_dist=x_dist, noise_std=0.5,
                                                     bounds=bounds)
                # take a subset of the biggest dataset for smaller datasets
                ds_input_dict[ds_size] = datasets.PointDataset(ds_size, gt_model,
                                                     x_dist=x_dist, noise_std=0.5,
                                                     bounds=bounds, data=(ds.X[0:ds_size], ds.Y[0:ds_size]))

            for specification in exp2_specs: # evaluate each given specification for in context models, useful for ablations

                ds_input = ds_input_dict[specification['train_specs']['dataset_size']]

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
                    posterior = in_context_model.eval_model.bayes_linear_posterior(ds_input.X, ds_input.Y)

                    predictions, params, _ = in_context_model.predict(torch.cat((ds_input.X, ds_input.Y), dim=-1).unsqueeze(0),
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

                    res_dict = {
                        "trial": i,
                        'gt': gt_model._get_name(),
                        'model_name': loss + " " + in_context_model.eval_model._get_name(),
                        # "mse_params_closed_form_gradient_descent": mse(closed_form_params, params).item(),
                        "mse_closed_form_gradient_descent": mse(closed_form_prediction, predictions).item(),
                        "mse_closed_form": mse(closed_form_prediction, ds_test.Y).item(),
                        "mse_gradient_descent": mse(predictions, ds_test.Y).item()
                    }

                    """if loss == 'backward-kl' or loss == 'forward-kl':
                        posterior = (posterior[0], posterior[1].squeeze(0))
                        mat = torch.eye(params[1].size(-1)).to(config.device) * params[1]
                        pred_dist = (params[0].squeeze(0), mat)
                        fw_kl = metrics.kl_mvn(pred_dist, posterior)
                        bw_kl = metrics.kl_mvn(posterior, pred_dist)
                        res_dict["fw_kl"] = fw_kl
                        res_dict["bw_kl"] = bw_kl
                        #res_dict["baseline_fwd"] = metrics.kl_mvn((torch.zeros_like(params[0]), torch.ones_like(torch.exp(params[1]))), posterior)
                        #res_dict["baseline_rev"] = metrics.kl_mvn(posterior, (torch.zeros_like(params[0]), torch.ones_like(torch.exp(params[1]))))
                    """
                    specification['results'][2].append(res_dict)


                    # TODO generate input dataset of 128 points randomly from input space

                    if plot:
                        os.makedirs(save_path + "/plots/", exist_ok=True)

                        Xplot = torch.linspace(-2, 2, 25).unsqueeze(1).to(config.device)
                        Xplot = torch.cat([Xplot, Xplot, Xplot], dim=-1)
                        Yplot = gt_model(Xplot)

                        # plotting is flawed, need to give different input to amortized model / cf
                        # sample an input dataset from ground truth
                        ds_input_plot = datasets.PointDataset(specification['train_specs']['dataset_size'], gt_model, x_dist=x_dist, noise_std=0.5, bounds=torch.tensor([[-2., 2.], [-2., 2.], [-2., 2.]]))

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

    if nop_specs:
        df = pd.DataFrame(nop_results)
        df_avg = df.groupby(['gt', 'model_name']).mean().reset_index()
        df_avg.to_csv(nop_specs['save_path'] + "/neuralop.csv", index=False)


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
        'min_lr': 1e-6,
        'weight_decay': 1e-5,
        'dataset_amount': 1000, #100000,
        'dataset_size': 128,
        'num_iters': 1,#1000000,
        'batch_size': 100,
        'valset_size': 1000, #10000,
        'normalize': False
    },
    'early_stopping_params': {
        'early_stopping_enabled': False,
        'patience': 10,
        'min_delta': 0.01,
        'load_best': True
    },
    'losses': losses,
    'save_path': './exp2_default',
    'save_all': False,
}
specs_3 = copy.deepcopy(default_specs)
specs_3['save_path'] = './exp2_uniform_fixed_normalize17112025'
specs_3['train_specs']['normalize'] = True
run_experiments([specs_3], nop_specs=None, x_dist='uniform_fixed')
