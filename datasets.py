import numpy as np
import torch
from torch.utils.data import Dataset
from sample import sample_normal


def sample_dataset(dataset_size, model, noise_std=0.1):
    """
    Sample a dataset of desired size.
    :param model: The model which is applied to generate y values from x values
    """

    # sample x values
    X = sample_normal((dataset_size, model.dx))

    # compute y values
    Y = model(torch.from_numpy(X).float())
    Y = Y.detach().numpy()

    # add noise
    Y = Y + sample_normal(shape=Y.shape, mean=0, std=noise_std)

    return X, Y


class PointDataset(Dataset):
    """
    Dataset class used for classical models, elements of the datasets are (x,y) pairs
    """

    def __init__(self, size, model, noise_std=0.1):
        self.model = model
        self.X, self.Y = sample_dataset(size, self.model, noise_std)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.from_numpy(self.Y[idx]).float()


class ContextDataset(Dataset):
    """
    Dataset class used to train amortized models, elements of the datasets are again
    whole datasets, i. e. a set of (x,y) pairs
    """

    def __init__(self, size, ds_size, model_class, dx, dy, noise_std=0.1, **kwargs):
        self.data = []
        self.params = []

        # Create 'size' amount of datasets which will make up the big context dataset
        for i in range(size):
            # Create new model which will be underlying this dataset
            model = eval(model_class)(dx, dy, kwargs['order'] if 'order' in kwargs else kwargs['dh'])
            X, Y = sample_dataset(ds_size, model, noise_std)
            self.data.append(torch.cat((X, Y), dim=1))
            self.params.append(model.get_W())

        # Store the actual datasets...
        self.data = torch.cat(self.data, dim=0)
        # ...and the parameters that were used to generate them
        self.params = torch.cat(self.params, dim=0)