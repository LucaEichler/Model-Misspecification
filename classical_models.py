import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import init

from datasets import sample_dataset
from sample import sample_normal


class NonLinear(nn.Module):
    def __init__(self, dx, dy, dh):
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.dh = dh
        self.linear1 = nn.Linear(dx, dh)
        self.linear2 = nn.Linear(dh, dy)
        self.relu = nn.ReLU()

        # init weights of network layers from standard normal distribution
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, mean=0.0, std=1.0)
                if m.bias is not None:
                    init.normal_(m.bias, mean=0.0, std=1.0)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

    def compute_loss(self, batch, loss_fns):
        X, Y = batch
        prediction = self(X)
        return loss_fns["MSE"](prediction, Y)


class NonLinearVariational(NonLinear):
    def __init__(self, dx, dy, dh=100):
        super().__init__(dx, dy, dh)
        self.var = nn.Parameter(torch.abs(torch.from_numpy(sample_normal(1)).float()))

    def forward(self, x, **kwargs):
        # Use reparameterization trick
        # each layer is of form Wx+b where W and b are now means
        # we can sample according to (W+eps1*var)x+(b+eps2*var) = Wx+b+eps1*var*x+eps2*var

        if kwargs.get('plot', False):
            return super().forward(x)


        eps1w = torch.randn_like(self.linear1.weight)
        eps1b = torch.randn_like(self.linear1.bias)

        eps2w = torch.randn_like(self.linear2.weight)
        eps2b = torch.randn_like(self.linear2.bias)

        layer1out = self.relu(self.linear1(x) + torch.matmul(x, (eps1w * self.var).T) + eps1b * self.var)
        layer2out = self.linear2(layer1out) + torch.matmul(layer1out, (eps2w * self.var).T) + eps2b * self.var
        return layer2out


    def compute_loss(self, batch, loss_fns):
        X, Y = batch

        prediction = self(X)
        L = self.dx*self.dh+self.dh+self.dh*self.dy+self.dy
        kl_div = 0.5 * ((L * (self.var - torch.log(self.var) - 1)) +
                        torch.sum(self.linear1.weight ** 2) + torch.sum(self.linear1.bias ** 2) +
                        torch.sum(self.linear2.weight ** 2) + torch.sum(self.linear2.bias ** 2))
        print("MSE: " + str(loss_fns["MSE"](prediction, Y)))
        print("KL: " + str(kl_div))
        return loss_fns["MSE"](prediction, Y) * 10 + kl_div


class Linear(nn.Module):
    def __init__(self, dx, dy, order=1):
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.order = order
        self.K = 1 + dx
        if self.order == 2:
            self.K = int(1 + dx + dx ** 2)

        # initialize parameters according to N(0,I)
        self.W = nn.Parameter(torch.from_numpy(sample_normal((self.dy, self.K))).float())

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

    def compute_loss(self, batch, loss_fns):
        X, Y = batch
        prediction = self(X)
        return loss_fns["MSE"](prediction, Y)


class LinearVariational(Linear):
    def __init__(self, dx, dy, order=1):
        super().__init__(dx, dy, order)

        # now the means and variance become the parameters, and the weights will be sampled during forward
        # TODO check if mus and var should be initialized to zero / one but I think random is correct?
        self.mus = nn.Parameter(torch.from_numpy(sample_normal((self.dy, self.K))).float())
        self.var = nn.Parameter(torch.abs(torch.from_numpy(sample_normal(1)).float()))

        # delete W (such that it is not a parameter anymore)
        del self.W

    def forward(self, x, **kwargs):
        # sample W, use "reparameterization trick"
        self.W = self.mus + self.var * torch.randn_like(self.mus)
        return super().forward(x)

    def compute_loss(self, batch, loss_fns):
        X, Y = batch

        prediction = self(X)
        kl_div = 0.5 * (self.dy * self.K * (self.var - torch.log(self.var) - 1) + torch.sum(self.mus ** 2))
        print("MSE: " + str(loss_fns["MSE"](prediction, Y)))
        print("KL: " + str(kl_div))
        return loss_fns["MSE"](prediction, Y) * 10 + kl_div


def test_linear():
    """
    create a random linear model and plot
    """

    X = torch.linspace(0, 1, 128).view(128, 1)

    X = torch.cat((X, X), dim=1)

    order_one = Linear(dx=2, dy=1, order=1)

    Y = order_one(X)

    # plots the y-values for the diagonal where (x0=x1)
    plt.plot(X.detach().numpy(), Y.detach().numpy())
    plt.show()


def test_linear():
    """
    create a random linear model and plot
    """

    X = torch.linspace(0, 1, 128).view(128, 1)

    X = torch.cat((X, X), dim=1)

    nonlin = NonLinear(dx=2, dy=1, dh=100)

    Y = nonlin(X)

    # plots the y-values for the diagonal where (x0=x1)
    plt.plot(X.detach().numpy(), Y.detach().numpy())
    plt.show()


def test_sample():
    """
    test the dataset sampling function
    :return:
    """
    model = NonLinear(dx=1, dy=1, dh=100)
    X, Y = sample_dataset(100, model)
    plt.scatter(X, Y)
    plt.show()
