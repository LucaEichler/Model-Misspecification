import numpy as np
import torch
from torch.utils.data import Dataset
from sample import sample_normal
from classical_models import Linear, NonLinear
from config import device


def sample_dataset(dataset_size, model, noise_std=0.0):
    """
    Sample a dataset of desired size.
    :param model: The model which is applied to generate y values from x values
    """

    # sample x values
    X = torch.randn(dataset_size, model.dx).to(device)

    # compute y values
    Y = model(X)

    # add noise
    noise = torch.randn_like(Y) * noise_std * (torch.max(Y)-torch.min(Y))
    Y = Y + noise

    return X, Y.detach()


class PointDataset(Dataset):
    """
    Dataset class used for classical models, elements of the datasets are (x,y) pairs
    """

    def __init__(self, size, model, noise_std):
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
                model = eval(model_class)(dx, dy, kwargs['order'] if 'order' in kwargs else kwargs['dh']).to(device)
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

