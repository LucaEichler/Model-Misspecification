import numpy as np
import torch
from torch.utils.data import Dataset

import config
from sample import sample_normal
from classical_models import Linear, NonLinear
from config import device
from test4 import NonLinearRegression


def sample_dataset(dataset_size, model, noise_std=0.0, new_sampling_method=True):
    """
    Sample a dataset of desired size.
    :param model: The model which is applied to generate y values from x values
    """
    if new_sampling_method:
        X = []
        for i in range(model.dx):
            bounds = torch.empty(2).uniform_(-10, 10)
            lo, hi = bounds.min(), bounds.max()

            xi = torch.empty(dataset_size).uniform_(lo, hi).unsqueeze(-1)
            X.append(xi)
        X = torch.cat(X, dim=-1)
    else:
        # sample x values
        X = torch.randn(dataset_size, model.dx).to(device)

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

    def __init__(self, size, model, noise_std=0.0):
        self.model = model.to(device)
        self.X, self.Y = sample_dataset(size, self.model, noise_std)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class ContextDataset(Dataset):
    """
    Dataset class used to train amortized models, elements of the datasets are again
    whole datasets, i. e. a set of (x,y) pairs
    """

    def __init__(self, size, ds_size, model_class, dx, dy, noise_std=0.0, **kwargs):
        self.data = []
        self.params = []

        with torch.no_grad():
            # Create 'size' amount of datasets which will make up the big context dataset
            for i in range(size):
                # Create new model which will be underlying this dataset
                model = eval(model_class)(dx, dy, **kwargs).to(device)
                X, Y = sample_dataset(ds_size, model, noise_std)
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

