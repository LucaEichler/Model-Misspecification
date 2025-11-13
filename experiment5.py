# Experiment for training amortized models to predict neural network parameters
# using parameters of trained networks as teacher
import numpy as np
import torch

import config
import datasets
import in_context_models
import main
import plotting
from main import train_in_context_models
import seed
from classical_models import Linear, NonLinear
from main import train

# generate mode to generate new data and write it to a file, as training neural networks takes a long time
# train mode for then using that data to train
mode = "generate"
dx = 3
dy = 1
dh = 100

filename = "exp5_params_uniform_fixed.csv"  # file to save training data in
filename_error = "./exp5_mse_uniform_fixed.csv" # file to save the MSE ("data quality") of the parameters
#filename_train = "./exp5_data_train.csv" # use different filename for train so that generate does not accidently write into it
filename_bounds = "./exp5_bounds_uniform_fixed.csv" # use different filename for train so that generate does not accidently write into it

test_set_size = 10000
dataset_size = 20000
num_iters = 1000000
gen_iterations = 100000 # how many parameters to generate
validation_frequency = 1000
normalize = False # use normalized data for training neural networks
x_dist = 'uniform_fixed'
weight_decay = 1e-5
lr = 0.01

early_stopping_params = {
        'early_stopping_enabled': True,
        'patience': 10,
        'min_delta': 0.01,
        'load_best': True
    }

if mode == "generate":
    if normalize:
        norm_fct = datasets.normalize
        norm_to_scale_fct = datasets.norm_to_scale
        renorm_fct = datasets.renorm
    else:
        def identity_norm(input):
            return input.X, input.Y, None, None
        def identity_rescale(input, scale):
            return input

        norm_fct = identity_norm
        norm_to_scale_fct = identity_rescale
        renorm_fct = identity_rescale

    for i in range(gen_iterations):
        gt_model = Linear(dx=dx, dy=dy, order=3, feature_sampling_enabled=True, nonlinear_features_enabled=True)

        if x_dist == 'uniform' or x_dist == 'uniform_fixed':
            bounds = datasets.gen_uniform_bounds(dx, x_dist)
        else:   bounds = torch.tensor([-2, 2, -2, 2, -2, 2]).reshape(dx, 2)

        test_set = datasets.PointDataset(size=test_set_size, model=gt_model, x_dist=x_dist, noise_std=0., bounds=bounds)

        ds = datasets.PointDataset(size=dataset_size, model=gt_model, x_dist=x_dist, noise_std=0.5, bounds=bounds)
        ds_val = datasets.PointDataset(size=dataset_size, model=gt_model, x_dist=x_dist, noise_std=0.5, bounds=bounds)

        x_norm, y_norm, x_scale, y_scale = norm_fct(ds)
        ds.X, ds.Y = x_norm, y_norm
        if normalize:
            ds_val.X, ds_val.Y = norm_to_scale_fct(ds_val.X, x_scale), norm_to_scale_fct(ds_val.Y, y_scale)

        model_nn = NonLinear(dx=dx, dy=dy, dh=dh)
        model_nn = train(model_nn, ds, valset=ds_val, valfreq=validation_frequency, iterations=num_iters, batch_size=100,
                         lr=lr, weight_decay=config.weight_decay_classical, use_wandb=False, save_path=None, early_stopping_params=early_stopping_params)

        gt_Y = gt_model(test_set.X)

        Y_pred_nn = model_nn(test_set.X)
        #Y_pred_nn = renorm_fct(model_nn(norm_to_scale_fct(test_set.X, x_scale)), y_scale)

        mse_nn = torch.mean((Y_pred_nn - gt_Y) ** 2)
        mse_nn_rel = torch.sum((Y_pred_nn - gt_Y) ** 2) / torch.sum(gt_Y ** 2)

        """# reject parameters with bad approximation
        if mse_nn < 0.1:"""
        W = model_nn.get_W()
        with open(filename, "a") as f:
            f.write(",".join(map(str, W.tolist())) + "\n")
        with open(filename_error, "a") as f:
            f.write((str(mse_nn_rel.item())) + "\n")
        if x_dist == 'uniform':
            with open(filename_bounds, "a") as f:
                f.write(",".join(map(str, bounds.flatten().tolist())) + "\n")


        # visualize in normalized
        plotting.plot_regression_on_dataset(test_set.Y, Y_pred_nn, "./plots/exp5_plot"+str(i))


        """
        start = bounds[:, 0]
        end = bounds[:, 1]

        # number of points along the line
        N = 25

        # linear interpolation for each coordinate
        t = torch.linspace(0., 1., N).unsqueeze(1)  # shape (N,1)
        line = start + t * (end - start)  # shape (N,3)

        Xplot = line.to(config.device)
        Yplot = norm_to_scale_fct(gt_model(Xplot), y_scale)

        Y_predplot = model_nn(norm_to_scale_fct(Xplot, x_scale))

        main.eval_plot_nn(str(i), Yplot, t, Y_predplot)
        """


elif mode == "train":
    #bounds = np.loadtxt(filename_bounds, delimiter=",")
    #bounds = torch.tensor(bounds, dtype=torch.float32).reshape(-1, dx, 2)

    data = np.loadtxt(filename, delimiter=",")
    tensors = torch.tensor(data, dtype=torch.float32)
    s = tensors.size(0)
    train_data = tensors[0:int(s*0.9), :]
    #train_bounds = bounds[0:8000, :, :]
    val_data = tensors[int(s*0.9):s, :]
    #val_bounds = bounds[8000:9000,:, :]


    batch_size = config.batch_size_in_context

    specification = {
        'transformer_arch': {
        'dT': 256,
        'num_heads': 4,
        'num_layers': 4,
        'output': 'attention-pool'
    },
        'train_specs': {
        'lr': 0.0001,
        'min_lr': 1e-6,
        'weight_decay': 1e-5,
        'dataset_amount': 100000,
        'dataset_size': 128,
        'num_iters': 100000,
        'batch_size': 100,
        'normalize': False
    },
        'save_path': './exp5_default',
        'losses': ['mle-dataset'],
        'early_stopping_params': {
            'early_stopping_enabled': False,
            'patience': 10,
            'min_delta': 0.01,
            'load_best': True
        },
    }

    model_name = 'NonLinear'
    model_kwargs = {'dh': config.dh}
    model_specs = [(model_name, model_kwargs)]

    dataset = datasets.ContextDataset(train_data.size(0), config.dataset_size_in_context, model_name, dx, dy, x_dist='uniform', params_list=train_data, **model_kwargs)
    valset = datasets.ContextDataset(val_data.size(0), config.dataset_size_in_context, model_name, dx, dy, x_dist='uniform', noise_std=config.noise_std, params_list=val_data, **model_kwargs)


    trained_in_context_models = train_in_context_models(dx = dx, dy = dy, transformer_arch = specification['transformer_arch'],
                                 x_dist = 'uniform', train_specs = specification['train_specs'],
                                 model_specs = model_specs, losses = specification['losses'], early_stopping_params = specification[
        'early_stopping_params'], save_path = specification['save_path'], datasets = (dataset, valset))


    _, model_trained = trained_in_context_models[0]


    tries = 10
    for i in range(tries):
        evalmodel = Linear(dx=dx, dy=dy, order=3, feature_sampling_enabled=False, nonlinear_features_enabled=True)

        # create new ground truth model for evaluation
        gt_model = eval(model_name)(dx=dx, dy=dy, **model_kwargs)

        Xplot = torch.linspace(-2, 2, 25).unsqueeze(1).to(config.device)
        Xplot = torch.cat([Xplot, Xplot, Xplot], dim=-1)
        Yplot = gt_model(Xplot)

        # plotting is flawed, need to give different input to amortized model / cf
        # sample an input dataset from ground truth
        ds_input_plot = datasets.PointDataset(dataset_size, gt_model, x_dist='uniform', noise_std=0.5,
                                              bounds=torch.tensor([[-2., 2.], [-2., 2.], [-2., 2.]]))

        Y_predplot, params_predplot = model_trained.predict(
            torch.cat((ds_input_plot.X, ds_input_plot.Y), dim=-1).unsqueeze(0),
            Xplot.unsqueeze(0))

        closed_form_params = evalmodel.closed_form_solution_regularized(ds_input_plot.X,
                                                                                          ds_input_plot.Y,
                                                                                          lambd=config.lambda_mle)
        Ypred_cf = evalmodel.forward(Xplot.unsqueeze(0), closed_form_params.unsqueeze(0))

        main.eval_plot(gt_model._get_name() + " " + str(i), model_trained.eval_model._get_name(),
                       gt_model, Xplot[:, 0], Y_predplot.squeeze(0), Ypred_cf)


