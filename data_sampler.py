import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn


def sample_params(shape):
    """
    Sample a desired amount of parameters from a Gaussian distribution with mean = 0 and var = 1
    """
    return np.random.normal(size=shape)


def sample_dataset(dataset_size, model, noise_var=0.1):
    """
    Sample a dataset of desired size.
    :param model: The model which is applied to generate y values from x values
    :return:
    """

    # sample x values
    X = sample_params((dataset_size, model.dx))

    # compute y values
    Y = model(torch.from_numpy(X).float())
    Y = Y.detach().numpy()

    # add noise
    Y = Y + np.random.normal(size=Y.shape, loc=0, scale=noise_var)

    return X, Y





class Linear(nn.Module):
    def __init__(self, dx, dy, order=1):
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.order = order
        self.K = 1 + dx
        if self.order is 2:
            self.K = int(1 + dx + dx ** 2)

        # initialize parameters according to N(0,I)
        self.W = nn.Parameter(torch.from_numpy(sample_params((self.dy, self.K)))).float()

    def forward(self, x):
        """

        :param x: (batch_size, dx)
        :return: (batch_size, dy)
        """
        batch_size = x.size(0)

        # bias
        ones = torch.ones((batch_size, 1))  # (batch_size, 1)

        # basis function vectors
        phi = torch.cat((ones, x), dim=1)  # (batch_size, dx+1)

        # calculate the outer product of each input vector with itself to get all second order basis functions
        if self.order == 2:
            outer_products = torch.einsum('ni,nj->nij', x, x)  # (batch size, dx, dx)
            outer_products = outer_products.view(outer_products.size(0),
                                                 outer_products.size(1) ** 2)  # (batch_size, dx**2)
            phi = torch.cat((phi, outer_products), dim=1)  # (batch_size, 1+dx+dx**2)

        # perform matrix multiplication (multiply the weight matrix W with the
        # basis function vector phi of dimensionality K)
        return torch.matmul(self.W, phi.T).T  # (batch_size, dy)


def test_linear():
    """
    create a random linear model of order one and order two and plot them
    """

    X = torch.linspace(0, 1, 128).view(128, 1)

    X = torch.cat((X, X), dim=1)

    order_one = Linear(dx=2, dy=1, order=1)

    Y = order_one(X)

    # plots the y-values for the diagonal where (x0=x1)
    plt.plot(X.detach().numpy(), Y.detach().numpy())
    plt.show()

def test_sample():
    model = Linear(dx=1, dy=1)
    X, Y = sample_dataset(100, model)
    plt.scatter(X,Y)
    plt.show()
