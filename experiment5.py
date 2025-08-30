# Experiment for training amortized models to predict neural network parameters
# using parameters of trained networks as teacher
import numpy as np
import torch

import config
import datasets
import in_context_models
from classical_models import Linear, NonLinear
from main import train

# generate mode to generate new data and write it to a file, as training neural networks takes a long time
# train mode for then using that data to train
mode = "train" # "train"
dx = 3
dy = 1
dh = 100

filename = "./exp5_data.csv" # file to save training data in TODO add file ending

test_set_size = 10000
dataset_size = 50000
num_iters = 1 #100000
gen_iterations = 10 # how many datasets to generate

if mode == "generate":
    for i in range(gen_iterations):
        gt_model = Linear(dx=dx, dy=dy, order=3, feature_sampling_enabled=True, nonlinear_features_enabled=True)
        test_set = datasets.PointDataset(size=test_set_size, model=gt_model, noise_std=0.5)

        ds = datasets.PointDataset(size=dataset_size, model=gt_model, noise_std=0.5)
        ds_val = datasets.PointDataset(size=dataset_size, model=gt_model, noise_std=0.5)

        model_nn = NonLinear(dx=dx, dy=dy, dh=100)
        model_nn = train(model_nn, ds, valset=ds_val, valfreq=1000, iterations=num_iters, batch_size=100,
                         lr=config.lr_classical, use_wandb=config.wandb_enabled)

        gt_Y = gt_model(test_set.X)
        Y_pred_nn = model_nn(test_set.X)

        mse_nn = torch.mean((Y_pred_nn - gt_Y) ** 2)

        # reject parameters with bad approximation
        if mse_nn < 0.1:
            W = model_nn.get_W()
            with open(filename, "a") as f:
                f.write(",".join(map(str, W.tolist())) + "\n")


elif mode == "train":
    data = np.loadtxt(filename, delimiter=",")
    tensors = torch.tensor(data, dtype=torch.float32)
    loss = "mle-params"

    model_name = 'NonLinear'
    model_kwargs = {'dh': config.dh}
    batch_size = config.batch_size_in_context

    model = in_context_models.InContextModel(dx, dy, 256, 4, 4, model_name, loss,
                                             **model_kwargs)
    dataset = datasets.ContextDataset(tensors.size(0), config.dataset_size_in_context, model_name, dx, dy, config.noise_std, params_list=tensors, **model_kwargs)
    valset = datasets.ContextDataset(1000, config.dataset_size_in_context, model_name, dx, dy, config.noise_std, **model_kwargs)
    model_trained = train(model, dataset, valfreq=500, valset=valset, iterations=num_iters, batch_size=batch_size,
                          lr=config.lr_in_context, use_wandb=config.wandb_enabled)


