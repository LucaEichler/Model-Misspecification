import numpy as np
import torch
from torch.utils.data import Dataset

import config
import in_context_models
from sample import sample_normal
from classical_models import Linear, NonLinear
from config import device
from test4 import NonLinearRegression


def gen_uniform_bounds(dim, min=-10., max=10.):
    bounds = torch.empty((dim,2))
    for i in range(dim):
        lohi = torch.empty(2).uniform_(-10, 10)
        lo, hi = lohi.min(), lohi.max()
        bounds[i,0], bounds[i,1] = lo, hi
    return bounds


def sample_dataset(dataset_size, model, x_dist, noise_std=0.0, bounds=None):
    """
    Sample a dataset.
    :param model: The model which is applied to generate y values from x values
    :param x_dist: distribution of input points, can be either gaussian or uniform
    :param bounds: if using uniform mode, specific bounds for each dimension may be provided here
    :return tuple of X and Y of dataset points
    """
    if x_dist == "uniform":
        X = []
        if bounds is None:
            bounds = gen_uniform_bounds(model.dx)
        for i in range(model.dx):
            lo, hi = bounds[i, 0], bounds[i, 1]

            xi = torch.empty(dataset_size).uniform_(lo, hi).unsqueeze(-1)
            X.append(xi)
        X = torch.cat(X, dim=-1).to(device)
    elif x_dist == "gaussian":
        # sample x values
        X = torch.randn(dataset_size, model.dx).to(device)
    else:
        raise ValueError(f"Wrong argument: {x_dist}. Expected 'uniform' or 'gaussian'.")

    # compute y values
    Y = model(X)

    noise = torch.normal(mean=0., std=noise_std, size=Y.shape, device = Y.device)
    # add noise (old)
    #noise = torch.randn_like(Y) * noise_std * (torch.max(Y)-torch.min(Y))
    Y = Y + noise

    return X.to(device), Y.detach()

class PointDataset(Dataset):
    """
    Dataset class used for classical models, elements of the datasets are (x,y) pairs
    """

    def __init__(self, size, model, x_dist, noise_std=0.0, bounds=None):
        self.model = model.to(device)
        self.X, self.Y = sample_dataset(size, self.model, x_dist, noise_std, bounds)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def min_max_norm(input):
    min, max = torch.min(input, dim=0, keepdim=True)[0], torch.max(input, dim=0, keepdim=True)[0]
    input_norm = (input-min)/(max-min)
    return input_norm, (min, max)

def norm_to_scale(input, scale):
    min, max = scale
    return (input-min)/(max-min)

def renorm(input, scale):
    min, max = scale
    return input*(max-min)+min
def normalize(dataset):
    X_norm, X_scale = min_max_norm(dataset.X)
    Y_norm, Y_scale = min_max_norm(dataset.Y)
    return X_norm, Y_norm, X_scale, Y_scale


class ContextDataset(Dataset):
    """
    Dataset class used to train amortized models, elements of the datasets are again
    whole datasets, i. e. a set of (x,y) pairs
    """

    def __init__(self, size, ds_size, model_class, dx, dy, x_dist, noise_std=0.0, params_list=None, bounds=None, **kwargs):
        self.data = []
        self.params = []

        W = None
        bound = None
        with torch.no_grad():
            # Create 'size' amount of datasets which will make up the big context dataset
            for i in range(size):
                if params_list is not None:
                    if bounds is not None:
                        bound = bounds[i, :, :]
                    W = params_list[i, :]
                    #TODO store and load bounds

                # Create new model which will be underlying this dataset
                model = eval(model_class)(dx, dy, init_W=W, **kwargs).to(device)
                X, Y = sample_dataset(ds_size, model, x_dist, noise_std, bound)
                self.data.append(torch.cat((X, Y), dim=1))
                self.params.append(model.get_W())


            # Store the actual datasets...
            self.data = torch.stack(self.data, dim=0)
            # ...and the parameters that were used to generate them
            self.params = torch.stack(self.params, dim=0)



    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :].to(device), self.params[idx].to(device)


class ContextDatasetAlternative(Dataset):
    """
    Inspired by Mittal et al.
    """

    def __init__(self, size, ds_size, model_class, dx, dy, batch_size, noise_std=0.0, **kwargs):
        self.set = NonLinearRegression(dim=1, min_len=49, max_len=50, min_param_len=10, max_param_len=100, h_dim=10,
                                  n_layers=1, act=torch.relu)
        self.batch_size = batch_size


    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        batch = self.set.get_batch(self.batch_size)
        x_train, y_train = batch[0]
        params, noiseorso = batch[2]
        return torch.cat([x_train.transpose(0,1), y_train.transpose(0,1)], dim=-1).to(device), torch.cat((params, torch.zeros_like(params[:, :1])),dim=-1).to(device)

    def get_batch(self):
        batch = self.set.get_batch(self.batch_size)
        x_train, y_train = batch[0]
        params, noiseorso = batch[2]
        return torch.cat([x_train.transpose(0, 1), y_train.transpose(0, 1)], dim=-1).to(device), torch.cat(
            (params, torch.zeros_like(params[:, :1])), dim=-1).to(device)


class ContextDatasetStream(Dataset):
    def __init__(self, batch_size, ds_size, model_class, dx, dy, noise_std=0.1, **kwargs):
        self.model_class = model_class
        self.ds_size = ds_size
        self.dx = dx
        self.dy = dy
        self.kwargs = kwargs
        self.noise_std = noise_std
        self.batch_size = batch_size

    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        data = []
        params = []

        # Create new model which will be underlying this dataset
        model = eval(self.model_class)(self.dx, self.dy, self.kwargs['order'] if 'order' in self.kwargs else self.kwargs['dh']).to(device)
        X, Y = sample_dataset(self.ds_size, model, self.noise_std)
        data.append(torch.cat((X, Y), dim=1))
        params.append(model.get_W())

        # Store the actual datasets...
        data = torch.cat(data, dim=0)
        # ...and the parameters that were used to generate them
        params = torch.cat(params, dim=0)
        return data.to(device), params.to(device)

