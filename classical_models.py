import warnings

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import init

import config
from config import dataset_size_classical, device, weight_decay_classical as weight_decay, basis_function_scaling_factors
from sample import sample_normal


class NonLinear(nn.Module):
    def __init__(self, dx, dy, dh, init_W=None):
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.dh = dh
        self.linear1 = nn.Linear(dx, dh)
        self.linear2 = nn.Linear(dh, dy)
        self.relu = nn.ReLU()
        self.to(device)

        if init_W is None:
            # init weights of network layers from standard normal distribution
            self._init_weights()
        else:
            # set weights to input parameter vector
            self.set_W(init_W)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, mean=0.0, std=1.0)
                if m.bias is not None:
                    init.normal_(m.bias, mean=0.0, std=1.0)

    def _init_weights_training(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)


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
        return torch.sum((prediction-Y)**2).mean()

    def count_params(self):
        return self.dx * self.dh + self.dh + self.dh * self.dy + self.dy

    def get_W(self):
        return torch.cat([
            self.linear1.weight.flatten(),
            self.linear1.bias.flatten(),
            self.linear2.weight.flatten(),
            self.linear2.bias.flatten()
        ])

    def set_W(self, W):  # TODO check correctness
        w1_dim = self.dh * self.dx
        b1_dim = self.dh
        w2_dim = self.dy * self.dh
        b2_dim = self.dy

        w1 = W[:w1_dim].view(self.dh, self.dx)
        b1 = W[w1_dim:w1_dim + b1_dim].view(self.dh)
        w2 = W[w1_dim + b1_dim:w1_dim + b1_dim + w2_dim].view(self.dy, self.dh)
        b2 = W[-b2_dim:].view(self.dy)

        with torch.no_grad():
            self.linear1.weight.copy_(w1)
            self.linear1.bias.copy_(b1)
            self.linear2.weight.copy_(w2)
            self.linear2.bias.copy_(b2)


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

        if self.training:
            samples = 1
        else: samples = 25

        stochastic_outputs = []

        for i in range(samples):
            eps1w = torch.randn_like(self.linear1.weight)
            eps1b = torch.randn_like(self.linear1.bias)

            eps2w = torch.randn_like(self.linear2.weight)
            eps2b = torch.randn_like(self.linear2.bias)

            layer1out = self.relu(self.linear1(x) + torch.matmul(x, (eps1w * torch.exp(self.logvar)).T) + eps1b * torch.exp(self.logvar))
            layer2out = self.linear2(layer1out) + torch.matmul(layer1out, (eps2w * torch.exp(self.logvar)).T) + eps2b * torch.exp(self.logvar)
            stochastic_outputs.append(layer2out)
        if self.training: return layer2out
        else: return torch.stack(stochastic_outputs, dim=0).mean(0)

    def compute_loss(self, batch):
        X, Y = batch

        prediction = self(X)
        L = self.dx * self.dh + self.dh + self.dh * self.dy + self.dy
        if self.training:
            kl_div = (0.5 * ((L *(torch.exp(self.logvar) - self.logvar - 1)) + torch.sum(self.get_W()**2)))
        else: kl_div = 0.
        noise_var = 0.25            # TODO make this configurable
        sse = (1./noise_var)*0.5 * torch.sum((prediction-Y)**2)*(1/0.5)

        return sse + kl_div


from itertools import combinations_with_replacement
import math
def num_monomials(d, k):
    return math.comb(d + k - 1, k)

def monomial_indices(d, k):
    combos = list(combinations_with_replacement(range(d), k))
    return torch.tensor(combos, dtype=torch.long)

class Linear(nn.Module):
    def __init__(self, dx, dy, order=1, init_W=None, nonlinear_features_enabled=False, feature_sampling_enabled=False):
        super().__init__()
        self.nonlinear_features_enabled = nonlinear_features_enabled
        self.feature_sampling_enabled=feature_sampling_enabled
        self.dx = dx
        self.dy = dy
        self.order = order
        self.K = 1 + dx
        if self.order >= 2:
            self.K = int(1 + dx + dx*(dx+1)/2)
            if self.order >=3:
                self.K += num_monomials(dx, 3)
        if self.nonlinear_features_enabled:
            self.K += dx*2*3
            self.K += 27

        if init_W is None:
            # initialize parameters according to N(0,I)
            self.W = nn.Parameter(torch.from_numpy(sample_normal((self.dy, self.K))).float())
        else:
            self.set_W(init_W)

        # In case of data generation, sample sparse weights for features, set rest to zero
        if self.feature_sampling_enabled:
            self.active_feature_indices = torch.empty(0, dtype=torch.long)
            if order == 1:
                L = torch.randint(1, 5, (1,)).item()
                self.active_feature_indices = torch.multinomial(torch.ones(4), L, replacement=False)
            elif order == 2:
                raise NotImplementedError  # For quadratic functions, we did not define this
            elif order == 3:
                o = 0
                if self.nonlinear_features_enabled:
                    Ldist_nonlin = 0.5**torch.arange(0, 46)


                    L_nonlin = torch.multinomial(Ldist_nonlin, 1, replacement=False).item()

                    idist_nonlin = torch.empty(45)
                    idist_nonlin[:18] = 1 / 36
                    idist_nonlin[18:] = 1 / 54

                    if L_nonlin != 0:
                        o = 1
                        self.active_feature_indices = torch.multinomial(idist_nonlin, L_nonlin, replacement=False) + 20

                Ldist_poly = (3**(-1/4))**torch.arange(0, 20 + o)  # We keep out the rest of the distribution as torch.multinomial does not need normalized distributions
                L_poly = torch.multinomial(Ldist_poly, 1, replacement=False).item() + 1 - o

                idist_poly = torch.empty(20)
                idist_poly[0:4] = 1 / 12
                idist_poly[4:10] = 1 / 18
                idist_poly[10:20] = 1 / 30

                if L_poly != 0: self.active_feature_indices = torch.cat((torch.multinomial(idist_poly, L_poly, replacement=False), self.active_feature_indices))  # TODO understand why we need gumbal trick and activate replacement if needed

            mask = torch.ones(self.K, dtype=torch.bool)
            mask[self.active_feature_indices] = False
            inactive_indices = torch.arange(self.K)[mask]

            with torch.no_grad():
                self.W[:, inactive_indices] = 0.


    def get_design_matrix(self, x, scales=None):
        batch_size = x.size(0)

        # bias
        ones = torch.ones((batch_size, 1)).to(self.sample_W().device)  # (batch_size, 1)

        # basis function vectors
        phi = torch.cat((ones, x), dim=1)  # (batch_size, dx+1)

        # calculate the outer product of each input vector with itself to get all second order basis functions
        if self.order >= 2:
            outer_products = torch.einsum('ni,nj->nij', x, x)  # (batch size, dx, dx)
            idxs = torch.triu_indices(outer_products.size(1), outer_products.size(2))
            outer_products = outer_products[:, idxs[0],
                             idxs[1]] *basis_function_scaling_factors['poly2'] #divide by 100 max range of x^2  # keep only upper triangular matrix to remove duplicate terms
            phi = torch.cat((phi, outer_products), dim=1)  # (batch_size, 1 + dx + dx*(dx+1)/2)

            if self.order >= 3:
                outer_products = torch.einsum('ni,nj,nk->nijk', x,x,x) # (batch size, dx, dx, dx)
                idxs = monomial_indices(self.dx, 3)
                outer_products = outer_products[:, idxs[:,0], idxs[:,1], idxs[:,2]] *basis_function_scaling_factors['poly3'] #divide by 1000 max range of x^3
                phi = torch.cat((phi, outer_products), dim=1)

        if self.nonlinear_features_enabled:
            # note that the next line also accounts for the normalized case
            products = torch.pi / 10 * torch.einsum('bi,j->bij', x if scales is None else x*(scales[:,0:3,1]-scales[:,0:3,0])+scales[:,0:3,0], torch.tensor([1,2,3], dtype=x.dtype, device = config.device)).view(batch_size, -1)
            cos_features = torch.cos(products) * basis_function_scaling_factors['fourier'] # divide by 2
            sin_features = torch.sin(products) * basis_function_scaling_factors['fourier'] # to get y-range of 1
            phi = torch.cat((phi, cos_features), dim=1)
            phi = torch.cat((phi, sin_features), dim=1)

            coords = torch.tensor([-7.5, 0., 7.5], device=config.device)
            a, b, c = torch.meshgrid(coords, coords, coords, indexing='ij')
            centers = torch.stack([a, b, c], dim=-1).view(-1, 3)
            s=torch.ones_like(centers)*4.5
            if scales is not None:  # In case of normalization, we are given scales to recalculate the centers and sizes
                centers = (centers[None, :, :]-scales[:,0:3,0])/(scales[:,0:3, 1]-scales[:,0:3,0])
                s = s[None, :, :]/(scales[:,0:3, 1]-scales[:,0:3,0])

            diff = x[:, None, :] - centers
            sq_dist = ((diff ** 2)/(2*(s**2))).sum(dim=2)
            gauss_features = torch.exp(-sq_dist) * basis_function_scaling_factors['exp']

            phi = torch.cat((phi, gauss_features), dim=1)

        return phi

    def sample_W(self):
        return self.W

    def forward(self, x, W=None, scales=None):
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


        phi = self.get_design_matrix(x, scales)

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

    def set_W(self, W):
        self.W = W.view(self.W.shape)

    def compute_loss(self, batch):
        X, Y = batch
        prediction = self(X)
        torch.sum(self.W.flatten() ** 2)
        return torch.sum((prediction-Y)**2).mean()

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

    def closed_form_solution_regularized(self, x, y, lambd):
        #TODO: Implement for variational methods too
        #TODO: Find out if intercept should be regularized or not
        if self.feature_sampling_enabled:
            warnings.warn("trying to compute a closed form solution for a model which"
                          "has feature sampling enabled - therefore it should be used for sampling data only")
        phi = self.get_design_matrix(x)
        return torch.linalg.solve(lambd*torch.eye(phi.size(-1), device=config.device) + phi.T @ phi, phi.T @ y)


    def _get_name(self):
        if self.order == 3:
            if self.nonlinear_features_enabled:
                return "Nonlinear"
            else: return "Polynomial"
        if self.order == 2:
            return "Quadratic"
        else: return "Linear"


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
        noise_var = 0.25    #TODO make configurable
        sse = 1./noise_var*torch.sum((prediction-Y)**2)
        return sse + kl_div

    def get_W(self):
        # TODO: this function will just regress towards the mean, so maybe just return the means?
        return self.mus.flatten()
        # perform MC sampling of parameters from posterior distribution
        W = torch.zeros_like(self.mus)
        for i in range(20):
            W += self.mus + torch.exp(self.logvar) * torch.randn_like(self.mus)
        W /= 20
        return W

    def _get_name(self):
        return super()._get_name() + "Variational"