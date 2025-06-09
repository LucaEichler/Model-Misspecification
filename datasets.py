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
