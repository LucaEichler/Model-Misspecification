# Experiment for training amortized models to predict neural network parameters
# using parameters of trained networks as teacher
import numpy as np
import torch

import config
import datasets
import in_context_models
import main
import seed
from classical_models import Linear, NonLinear
from main import train

# generate mode to generate new data and write it to a file, as training neural networks takes a long time
# train mode for then using that data to train
mode = "train" # "generate"
dx = 3
dy = 1
dh = 100

filename = "exp5_data.csv"  # file to save training data in
filename_error = "./exp5_mse.csv" # file to save the MSE ("data quality") of the parameters
filename_train = "./exp5_data_train.csv" # use different filename for train so that generate does not accidently write into it

test_set_size = 10000
dataset_size = 20000
num_iters = 1000000
gen_iterations = 1000 # how many parameters to generate
validation_frequency = 1000
normalize = False # use normalized data for training neural networks


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
        bounds = datasets.gen_uniform_bounds(dx)

        test_set = datasets.PointDataset(size=test_set_size, model=gt_model, x_dist='uniform', noise_std=0., bounds=bounds)

        ds = datasets.PointDataset(size=dataset_size, model=gt_model, x_dist='uniform', noise_std=0.5, bounds=bounds)
        ds_val = datasets.PointDataset(size=dataset_size, model=gt_model, x_dist='uniform', noise_std=0.5, bounds=bounds)

        x_norm, y_norm, x_scale, y_scale = norm_fct(ds)
        ds.X, ds.Y = x_norm, y_norm
        ds_val.X, ds_val.Y = norm_to_scale_fct(ds_val.X, x_scale), norm_to_scale_fct(ds_val.Y, y_scale)

        model_nn = NonLinear(dx=dx, dy=dy, dh=dh)
        model_nn = train(model_nn, ds, valset=ds_val, valfreq=validation_frequency, iterations=num_iters, batch_size=100,
                         lr=config.lr_classical, use_wandb=config.wandb_enabled)

        gt_Y = gt_model(test_set.X)
        Y_pred_nn = renorm_fct(model_nn(norm_to_scale_fct(test_set.X, x_scale)), y_scale)

        mse_nn = torch.mean((Y_pred_nn - gt_Y) ** 2)
        mse_nn_rel = torch.sum((Y_pred_nn - gt_Y) ** 2) / torch.sum(gt_Y ** 2)

        """# reject parameters with bad approximation
        if mse_nn < 0.1:"""
        W = model_nn.get_W()
        with open(filename, "a") as f:
            f.write(",".join(map(str, W.tolist())) + "\n")
        with open(filename_error, "a") as f:
            f.write((str(mse_nn_rel.item())) + "\n")


        # visualize in normalized space

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



elif mode == "train":
    data = np.loadtxt(filename_train, delimiter=",")
    tensors = torch.tensor(data, dtype=torch.float32)
    loss = "mle-params"

    model_name = 'NonLinear'
    model_kwargs = {'dh': config.dh}
    batch_size = config.batch_size_in_context

    model = in_context_models.InContextModel(dx, dy, 256, 4, 4, model_name, loss,
                                             **model_kwargs)
    dataset = datasets.ContextDataset(tensors.size(0), config.dataset_size_in_context, model_name, dx, dy, x_dist='uniform', noise_std=config.noise_std, params_list=tensors, **model_kwargs)
    valset = datasets.ContextDataset(1000, config.dataset_size_in_context, model_name, dx, dy, x_dist='uniform', noise_std=config.noise_std, **model_kwargs) #TODO: random split into train val test
    model_trained = train(model, dataset, valfreq=500, valset=valset, iterations=num_iters, batch_size=batch_size,
                          lr=config.lr_in_context, use_wandb=config.wandb_enabled)


