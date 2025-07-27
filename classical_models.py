import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import init
from config import dataset_size_classical, device, weight_decay_classical as weight_decay
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
        #init.zeros_(self.linear2.bias) #TODO keep or remove?

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, mean=0.0, std=1.0)
                if m.bias is not None:
                    init.normal_(m.bias, mean=0.0, std=1.0)

    def forward(self, x, W=None):
        # TODO: Check for correctness
        batched = W is not None
        if batched:
            B = W.size(0)

            # Compute sizes
            w1_dim = self.dh * self.dx
            b1_dim = self.dh
            w2_dim = self.dy * self.dh
            b2_dim = self.dy

            # Split
            w1 = W[:, :w1_dim].view(B, self.dh, self.dx)
            b1 = W[:, w1_dim:w1_dim + b1_dim].view(B, self.dh)
            w2 = W[:, w1_dim + b1_dim:w1_dim + b1_dim + w2_dim].view(B, self.dy, self.dh)
            b2 = W[:, -b2_dim:].view(B, self.dy)

            x1 = torch.bmm(w1, x.transpose(1, 2).to(device)).transpose(1, 2) + b1.unsqueeze(1)
            x1 = torch.relu(x1)
            return torch.bmm(w2, x1.transpose(1, 2).to(device)).transpose(1, 2) + b2.unsqueeze(1)

        return self.linear2(self.relu(self.linear1(x)))

    def compute_loss(self, batch):
        X, Y = batch
        prediction = self(X)
        if self.training:
            l2_penalty = weight_decay * torch.sum(self.get_W() ** 2)
        else: l2_penalty = 0.
        return torch.sum((prediction-Y)**2).mean() + l2_penalty

    def count_params(self):
        return self.dx * self.dh + self.dh + self.dh * self.dy + self.dy

    def get_W(self):
        return torch.cat([
            self.linear1.weight.flatten(),
            self.linear1.bias.flatten(),
            self.linear2.weight.flatten(),
            self.linear2.bias.flatten()
        ])

    def plot_eval(self, gt_model):
        # we need the gt params to plot
        X = torch.linspace(0, 1, 128).unsqueeze(1)
        num_samples = 20
        Y_pred = self(X)
        for i in range(1, num_samples):
            Y_pred += self(X)
        Y_pred = Y_pred / num_samples
        Y_gt = gt_model(X)
        plt.plot(X.detach().numpy(), Y_pred.detach().numpy())
        plt.plot(X.detach().numpy(), Y_gt.detach().numpy())
        plt.show()

class NonLinearVariational(NonLinear):
    def __init__(self, dx, dy, dh=100):
        super().__init__(dx, dy, dh)
        self.logvar = nn.Parameter(torch.tensor(-2.3))

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

        if self.training:
            layer1out = self.relu(self.linear1(x) + torch.matmul(x, (eps1w * torch.exp(self.logvar)).T) + eps1b * torch.exp(self.logvar))
            layer2out = self.linear2(layer1out) + torch.matmul(layer1out, (eps2w * torch.exp(self.logvar)).T) + eps2b * torch.exp(self.logvar)
        else:
            layer1out = self.relu(self.linear1(x))
            layer2out = self.linear2(layer1out)
        return layer2out

    def compute_loss(self, batch):
        X, Y = batch

        prediction = self(X)
        L = self.dx * self.dh + self.dh + self.dh * self.dy + self.dy
        if self.training:
            kl_div = 0.5 * ((L * (torch.exp(self.logvar) - self.logvar - 1)) +
                            torch.sum(self.linear1.weight ** 2) + torch.sum(self.linear1.bias ** 2) +
                            torch.sum(self.linear2.weight ** 2) + torch.sum(self.linear2.bias ** 2))
        else: kl_div = 0.
        sse = torch.sum((prediction-Y)**2)
        sse = sse / X.size(0) * dataset_size_classical  # normalize with dataset size

        return sse + kl_div


class Linear(nn.Module):
    def __init__(self, dx, dy, order=1):
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.order = order
        self.K = 1 + dx
        if self.order == 2:
            self.K = int(1 + dx + dx*(dx+1)/2)

        # initialize parameters according to N(0,I)
        self.W = nn.Parameter(torch.from_numpy(sample_normal((self.dy, self.K))).float())


    def get_design_matrix(self, x):
        batch_size = x.size(0)

        # bias
        ones = torch.ones((batch_size, 1)).to(self.sample_W().device)  # (batch_size, 1)

        # basis function vectors
        phi = torch.cat((ones, x), dim=1)  # (batch_size, dx+1)

        # calculate the outer product of each input vector with itself to get all second order basis functions
        if self.order == 2:
            # TODO check if 2nd order term symmetries removed correctly
            outer_products = torch.einsum('ni,nj->nij', x, x)  # (batch size, dx, dx)
            idxs = torch.triu_indices(outer_products.size(1), outer_products.size(2))
            outer_products = outer_products[:, idxs[0],
                             idxs[1]]  # keep only upper triangular matrix to remove duplicate terms
            phi = torch.cat((phi, outer_products), dim=1)  # (batch_size, 1 + dx + dx*(dx+1)/2)

        return phi

    def sample_W(self):
        return self.W

    def forward(self, x, W=None):
        """

        :param x: (batch_size, dx) or (num_datasets, num_points, dx) if batched
        :param W: use this if batched multiplication is required
        :return: (batch_size, dy)
        """
        x = x.to(self.sample_W().device)
        batch_size = x.size(0)

        batched = W is not None
        if batched:
            W = W.view(W.size(0), self.dy, self.K)
            batch_size = x.size(0) * x.size(1)
            prev_size = x.size(1)
            x = x.view(batch_size, x.size(2))


        phi = self.get_design_matrix(x)

        # bias
        ones = torch.ones((batch_size, 1)).to(device)  # (batch_size, 1)

        if batched:
            phi = phi.view(W.size(0), prev_size, phi.size(1)).to(device)
            return torch.bmm(W, phi.transpose(1, 2)).transpose(1, 2)

        # perform matrix multiplication (multiply the weight matrix W with the
        # basis function vector phi of dimensionality K)
        return torch.matmul(self.sample_W(), phi.T).T  # (batch_size, dy)

    def count_params(self):
        return self.W.numel()

    def get_W(self):
        return self.W.flatten()

    def compute_loss(self, batch):
        X, Y = batch
        prediction = self(X)
        if self.training: l2_penalty = weight_decay * torch.sum(self.W.flatten() ** 2)
        else: l2_penalty = 0.
        return torch.sum((prediction-Y)**2).mean() + l2_penalty

    def plot_eval(self, gt_model):
        # we need the gt params to plot
        X = torch.linspace(0, 1, 128).unsqueeze(1)
        num_samples = 20
        Y_pred = self(X)
        for i in range(1, num_samples):
            Y_pred += self(X)
        Y_pred = Y_pred / num_samples
        Y_gt = gt_model(X)
        plt.plot(X.detach().numpy(), Y_pred.detach().numpy())
        plt.plot(X.detach().numpy(), Y_gt.detach().numpy())
        plt.show()

    def closed_form_solution(self, x, y):
        #TODO: Implement for variational methods too
        phi = self.get_design_matrix(x)
        return torch.linalg.pinv(phi) @ y

    def _get_name(self):
        return super()._get_name()+str(self.order)


class LinearVariational(Linear):
    def __init__(self, dx, dy, order=1):
        super().__init__(dx, dy, order)

        # now the means and variance become the parameters, and the weights will be sampled during forward
        # TODO check if mus and var should be initialized to zero / one but I think random is correct?
        self.mus = nn.Parameter(torch.from_numpy(sample_normal((self.dy, self.K))).float())
        self.logvar = nn.Parameter(torch.tensor(-2.3).float())

        # delete W (such that it is not a parameter anymore)
        del self.W


    def sample_W(self):
        # sample W, use "reparameterization trick"
        if self.training:
            return self.mus + torch.exp(self.logvar) * torch.randn_like(self.mus)
        else:
            return self.mus

    def forward(self, x):


        return super().forward(x)

    def compute_loss(self, batch):
        X, Y = batch

        prediction = self(X)
        if self.training:
            kl_div = 0.5 * (self.dy * self.K * (torch.exp(self.logvar) - self.logvar - 1) + torch.sum(self.mus ** 2))
        else: kl_div = 0.
        sse = torch.sum((prediction-Y)**2)
        sse = sse / X.size(0) * dataset_size_classical  # normalize with dataset size
        return sse + kl_div

    def get_W(self):
        # TODO: this function will just regress towards the mean, so maybe just return the means?
        # perform MC sampling of parameters from posterior distribution
        W = torch.zeros_like(self.mus)
        for i in range(20):
            W += self.mus + torch.exp(self.logvar) * torch.randn_like(self.mus)
        W /= 20
        return W