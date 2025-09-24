
import numpy as np
from scipy.stats import t
import pandas as pd
import torch

import metrics
import seed
from main import train_in_context_models, train_classical_models, eval_plot_nn
from classical_models import Linear, NonLinear
from config import device

seed.set_seed(0)

def se(x):
    return x.std(ddof=1) / np.sqrt(len(x))

def ci95(x):
    n = len(x)
    std = x.std(ddof=1)
    se_val = std / np.sqrt(n)
    t_val = t.ppf(0.975, df=n - 1)
    return t_val * se_val


amortized_specs = {
        'transformer_arch':
            {
                'dT': 256,
                'num_heads': 4,
                'num_layers': 4,
                'output': 'attention-pool'
            },
        'train_specs':
            {
                'dataset_size': 128,
                'lr': 0.0001,
                'min_lr': 1e-6,
                'weight_decay': 1e-5,
                'dataset_amount': 100000,
                'num_iters': 100000,
                'batch_size': 100,
                'valset_size': 10000,
                'normalize': False
            },
        'early_stopping_params': {
            'early_stopping_enabled': True,
            'patience': 10,
            'min_delta': 0.01,
            'load_best': False
        },
}
classical_specs = {
    'dataset_size': 128,
    'weight_decay': 1e-5,
    'lr': 0.01,
    'num_iters': 100000,
    'early_stopping_params': {
            'early_stopping_enabled': True,
            'patience': 10,
            'min_delta': 0.01,
            'load_best': False
        },
    'batch_size': 100
}
default_specs = {
    'losses': ['mle-params', 'mle-dataset', 'forward-kl', 'backward-kl'],
    'save_path': './exp1_default',
    'dh': 10,
    'classical_model_specs': classical_specs,
    'amortized_model_specs': amortized_specs,
    'noise_std': 0.5,
    'trials': 1
}



def run_experiments(exp1_specs):
    for specification in exp1_specs:
        save_path = specification['save_path']

        model_specs = [('Linear', {'order': 1}), ('Linear', {'order': 2}), ('NonLinear', {'dh': specification['dh']})]

        linear_datasets, linear_2_datasets, nonlinear_datasets = train_classical_models(dx=1, dy=1, dh=specification['dh'], specs=specification['classical_model_specs'], x_dist='gaussian', tries=specification['trials'], noise_std=specification['noise_std'])

        trained_in_context_models = train_in_context_models(dx=1, dy=1, x_dist='gaussian', transformer_arch=specification['amortized_model_specs']['transformer_arch'], train_specs=specification['amortized_model_specs']['train_specs'], noise_std=specification['noise_std'], model_specs=model_specs, losses=specification['losses'], early_stopping_params=specification['amortized_model_specs']['early_stopping_params'], save_path=specification['save_path'], save_all=False)


        #X = torch.linspace(-5, 5, 128).unsqueeze(1)  # 128 equally spaced evaluation points between -1 and 1 - should we instead take a normally distributed sample here every time?
        X = torch.randn(128).unsqueeze(1).to(device) #TODO 128 points is a bit few, increase

        mse_results = []
        mse_params_results = []
        for model_type in [linear_datasets, linear_2_datasets, nonlinear_datasets]:
            j=1
            for elem in model_type:
                gt = elem[0]
                classical_models_trained = elem[2]

                Y = gt(X) # ground truth output to compare with

                for i in range(0, len(classical_models_trained)):

                    # we only can compare the parameters if there is no misspecification
                    if isinstance(classical_models_trained[i], type(gt)):
                        if not isinstance(classical_models_trained[i], Linear) or classical_models_trained[i].order == gt.order:
                            mse_params = metrics.mse(classical_models_trained[i].get_W(), gt.get_W())
                            mse_params_results.append({'gt': gt._get_name(), 'model_name': classical_models_trained[i]._get_name(),
                                                'mse_params': mse_params.item()})

                    classical_models_trained[i].eval()
                    Y_pred = classical_models_trained[i](X)
                    mse = metrics.mse(Y, Y_pred)
                    mse_results.append({'gt': gt._get_name(), 'model_name': classical_models_trained[i]._get_name(), 'mse': mse.item()})
                    #eval_plot(gt._get_name()+" "+str(j), classical_models_trained[i]._get_name(), gt, X, Y_pred)

                for trained_in_context_model in trained_in_context_models:

                    trained_in_context_model[1].to(device)
                    Y_pred, means_pred = trained_in_context_model[1].predict(torch.cat((elem[1].X, elem[1].Y), dim=-1).unsqueeze(0), X.unsqueeze(0))

                    if isinstance(trained_in_context_model[1].eval_model, type(gt)):
                        if not isinstance(trained_in_context_model[1].eval_model, Linear) or trained_in_context_model[1].eval_model.order == gt.order:
                            mse_params = metrics.mse(means_pred.squeeze(0), gt.get_W())
                            mse_params_results.append({'gt': gt._get_name(), 'model_name': trained_in_context_model[0]+" "+trained_in_context_model[1].eval_model._get_name(), 'mse_params': mse_params.item()})

                    eval_plot_nn(gt._get_name()+" "+str(j) + trained_in_context_model[0]+" "+trained_in_context_model[1].eval_model._get_name(), gt(X), X, Y_pred)

                    mse = metrics.mse(Y, Y_pred)

                    mse_results.append({'gt': gt._get_name(), 'model_name': trained_in_context_model[0]+" "+trained_in_context_model[1].eval_model._get_name(), 'mse': mse.item()})
                j=j+1
        df = pd.DataFrame(mse_results)
        df_params = pd.DataFrame(mse_params_results)

        # average over similar columns to compute mean performance across datasets
        df_avg = df.groupby(['gt', 'model_name'])['mse'].agg(
            mean_mse='mean',
            std_mse='std',
            se=se,
            ci=ci95
        ).reset_index()
        df_avg_params = df_params.groupby(['gt', 'model_name'], as_index=False)['mse_params'].mean()

        # Save to disk
        df_avg.to_csv(save_path+"/experiment1_results.csv", index=False)
        df_avg_params.to_csv(save_path+"/experiment1_params_results.csv", index=False)



specification = default_specs
specification['save_path'] = "./exp1"
run_experiments([specification])