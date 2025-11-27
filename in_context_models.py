from itertools import combinations_with_replacement

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

import config
from classical_models import Linear, NonLinear, monomial_indices
from config import device, weight_decay_in_context as weight_decay


def count_and_print_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)

def renormalize(y_norm, scales):
    # expect (batch_size, dy)
    y_min = scales[:, -1, 0][:, None, None]
    y_max = scales[:, -1, 1][:, None, None]
    y_renorm = (y_norm * (y_max-y_min))+y_min
    return y_renorm

def normalize_input(x):
    # Normalize each dimension
    x_norm = x.clone()
    scales = torch.zeros((x.size(1), x.size(2), 2), device=device)
    for i in range(x.size(-1)):  # potentially range(x.size(-1)-1) so that y is not normalized
        x_min = torch.min(x[:, :, i], dim=0, keepdim=True).values
        x_max = torch.max(x[:, :, i], dim=0, keepdim=True).values
        x_norm[:, :, i] = (x[:,:,i]-x_min)/(x_max-x_min)
        scales[:, i, 0]=x_min
        scales[:, i, 1]=x_max
    return x_norm, scales

def normalize_params(params, scales):  # params of shape (batch_size, num_params) #TODO as params are usually sparse, only compute contributions for active terms
    cub = config.basis_function_scaling_factors['poly3']
    quad = config.basis_function_scaling_factors['poly2']
    lin = config.basis_function_scaling_factors['lin']

    normalized_params = torch.zeros_like(params, dtype=params.dtype)
    y_min, y_max = scales[:, -1, 0], scales[:, -1, 1]
    x_min, x_max = scales[:, 0:3, 0], scales[:, 0:3, 1]

    normalized_params[:, 0] += params[:, 0] # add original bias

    normalized_params[:, 0] -= y_min #TODO check if this correct normalization or we need /ymax-ymin

    # linear terms
    normalized_params[:, 1:4]+=params[:, 1:4]*(x_max-x_min)
    normalized_params[:, 0] += torch.sum(params[:, 1:4]*x_min, dim=-1) * lin

    if params.size(-1) > 4:

        # quadratic terms
        outer_products = torch.einsum('ni,nj->nij', (x_max-x_min), (x_max-x_min))  # (batch size, dx, dx)
        idxs = torch.triu_indices(outer_products.size(1), outer_products.size(2))
        outer_products = outer_products[:, idxs[0],
                         idxs[1]]

        normalized_params[:, 4:10] += params[:, 4:10]*outer_products

        combos = list(combinations_with_replacement(range(3), 2))
        for i in range(len(combos)):
            normalized_params[:, combos[i][0]+1] += params[:, 4 + i] * (x_max - x_min)[:, combos[i][0]] * x_min[:, combos[i][1]] * (quad/lin)
            normalized_params[:, combos[i][1]+1] += params[:, 4 + i] * (x_max - x_min)[:, combos[i][1]] * x_min[:, combos[i][0]] * (quad/lin)

        # bias for quadratic terms
        outer_products = torch.einsum('ni,nj->nij', x_min, x_min)  # (batch size, dx, dx)
        idxs = torch.triu_indices(outer_products.size(1), outer_products.size(2))
        outer_products = outer_products[:, idxs[0],
                         idxs[1]]

        normalized_params[:, 0] += torch.sum(params[:, 4:10]*outer_products, dim=-1) * quad

        # cubic terms
        outer_products = torch.einsum('ni,nj,nk->nijk', (x_max-x_min), (x_max-x_min), (x_max-x_min))  # (batch size, dx, dx)
        idxs = monomial_indices(3, 3)
        outer_products = outer_products[:, idxs[:, 0], idxs[:, 1], idxs[:, 2]]

        normalized_params[:, 10:20] += params[:, 10:20]*outer_products

        # lookup table #TODO global variable
        quad_combos = list(combinations_with_replacement(range(3), 2))
        quad_index = {c: i for i, c in enumerate(quad_combos)}

        combos = list(combinations_with_replacement(range(3), 3))
        for i in range(len(combos)): #TODO add simple check if param is active before calculation
            (a,b,c) = combos[i]

            normalized_params[:, 4+quad_index[tuple(sorted((a, b)))]] += params[:, 10+i]*(x_max-x_min)[:, a]*(x_max-x_min)[:, b]*x_min[:, c] * (cub/quad)
            normalized_params[:, 4+quad_index[tuple(sorted((a, c)))]] += params[:, 10+i]*(x_max-x_min)[:, a]*(x_max-x_min)[:, c]*x_min[:, b] * (cub/quad)
            normalized_params[:, 4+quad_index[tuple(sorted((b, c)))]] += params[:, 10+i]*(x_max-x_min)[:, b]*(x_max-x_min)[:, c]*x_min[:, a] * (cub/quad)

            normalized_params[:, 1+a] += params[:, 10+i]*(x_max-x_min)[:, a]*x_min[:,b]*x_min[:,c] * (cub/lin)
            normalized_params[:, 1+b] += params[:, 10+i]*(x_max-x_min)[:, b]*x_min[:,a]*x_min[:,c] * (cub/lin)
            normalized_params[:, 1+c] += params[:, 10+i]*(x_max-x_min)[:, c]*x_min[:,b]*x_min[:,a] * (cub/lin)

        outer_products = torch.einsum('ni,nj,nk->nijk', x_min, x_min, x_min)  # (batch size, dx, dx)
        outer_products = outer_products[:, idxs[:, 0], idxs[:, 1], idxs[:, 2]]

        normalized_params[:,0]+=torch.sum(params[:, 10:20]*outer_products, dim=-1) * cub

    if params.size(-1) > 20:
        normalized_params[:,20:65] = params[:,20:65]

    normalized_params /= (y_max - y_min).unsqueeze(-1)

    return normalized_params


def normalize_to_scales(x, scales):
    x_norm = x.clone()
    for i in range(x.size(-1)):
        x_min = scales[:, i, 0][:,None]
        x_max = scales[:, i, 1][:,None]
        x_norm[:, :, i] = (x[:,:,i]-x_min)/(x_max-x_min)

    return x_norm

class TransformerAggregateOutput(nn.Module):
    def __init__(self, dx, dy, dOut, dT, num_heads, num_layers, normalize):
        super().__init__()
        self.normalize = normalize
        self.dIn = dx + dy
        self.dOut = dOut
        self.dT = dT

        self.learnable_query = nn.Parameter(torch.randn(1, 1, dT))
        self.multihead_attention = nn.MultiheadAttention(embed_dim=dT, num_heads=1)

        # This layer maps the input of dim dx+dy to the desired Transformer dimensionality dT
        self.encoder = nn.Linear(self.dIn, dT)

        # This layer maps from the Transformer output of dim dT to the desired output dim dT
        #self.decoder = nn.Sequential(nn.Linear(dT, dT*2), nn.ReLU(), nn.Linear(dT*2,dOut))
        self.decoder = nn.Linear(dT,dOut)

        self.scale_encoder = nn.Linear(2, dT)

        # init the actual Transformer
        tsf_layer = nn.TransformerEncoderLayer(d_model=dT, nhead=num_heads, dim_feedforward=4 * dT, batch_first=False)
        self.transformer_model = nn.TransformerEncoder(tsf_layer, num_layers=num_layers, enable_nested_tensor=False)


    def forward(self, x):
        scales = None
        if self.normalize:
            x_norm, scales = normalize_input(x)
            x = x_norm

        # x is of shape (seq_length, batch_size, dx)
        x_emb = self.encoder(x)

        # forward through transformer
        tf_out = self.transformer_model(x_emb)

        q = self.learnable_query.repeat(1, x.size(1), 1)
        attn_output, _ = self.multihead_attention(query=q, key=tf_out, value=tf_out)

        # map to desired output dimension & return
        return self.decoder(attn_output.squeeze(0)), scales



class Transformer2(nn.Module):
    def __init__(self, dx, dy, dOut, dT, num_heads, num_layers):
        super().__init__()
        self.normalize = True
        self.dIn = dx + dy
        self.dOut = dOut
        self.dT = dT

        # This layer maps the input of dim dx+dy to the desired Transformer dimensionality dT
        self.encoder = nn.Linear(self.dIn, dT)

        # This layer maps from the Transformer output of dim dT to the desired output dim dT
        self.decoder = nn.Linear(dT, dOut)

        self.scale_encoder = nn.Linear(2, dT)

        # init the actual Transformer
        tsf_layer = nn.TransformerEncoderLayer(d_model=dT, nhead=num_heads, dim_feedforward=4 * dT, batch_first=False)
        self.transformer_model = nn.TransformerEncoder(tsf_layer, num_layers=num_layers, enable_nested_tensor=False)

        # The CLS token which will contain the result after the transformer is applied
        self.CLS = nn.Parameter(torch.zeros(1, 1, dT))
        nn.init.xavier_uniform_(self.CLS)

    def forward(self, x):
        scales = None
        if self.normalize:
            x_norm, scales = normalize_input(x)
            x = x_norm

        # x is of shape (seq_length, batch_size, dx)
        x_emb = self.encoder(x)

        # repeat the CLS token to match the batch size
        cls = self.CLS.repeat(1, x.shape[1], 1)

        # concatenate (prepend) CLS to the input batch
        tf_in = torch.cat((cls, x_emb), dim=0)

        # forward through transformer
        tf_out = self.transformer_model(tf_in)

        # map to desired output dimension (only take output at CLS token) & return
        return self.decoder(tf_out[0, :, :]), scales


class InContextModel(nn.Module):
    def __init__(self, dx, dy, transformer_arch, output_model, loss, normalize=True, **kwargs):
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.loss = loss
        self.normalize = normalize

        # Create model which will be used for evaluation and freeze its parameters
        # With the batched forward, freezing should not be needed, maybe remove later
        self.eval_model = eval(output_model)(dx, dy, **kwargs).to(device)
        for param in self.eval_model.parameters():
            param.requires_grad = False

        # output of transformer should be of the size of the parameter vector for a model
        # (or mean and variance for bayesian methods i.e. *2)
        dOut = self.eval_model.count_params()
        if self.loss in ['forward-kl', 'backward-kl']:
            dOut *=2
        if transformer_arch['output'] == 'attention-pool':
            self.transformer = TransformerAggregateOutput(dx, dy, dOut, transformer_arch['dT'], transformer_arch['num_heads'], transformer_arch['num_layers'], normalize)
        elif transformer_arch['output'] == 'cls':
            self.transformer = Transformer2(dx, dy, dOut, transformer_arch['dT'], transformer_arch['num_heads'], transformer_arch['num_layers'])

        count_and_print_params(self)

    def forward(self, x):
        # Transformer directly maps to parameters
        return self.transformer(x)


    def compute_loss(self, batch):
        loss, *_ = self.compute_forward(batch)
        return loss


    def predict(self, context_datasets, input):
        # Given a dataset, the transformer predicts some parameters
        # Transpose such that sequence length is first dimension
        pred_params, scales = self(context_datasets.transpose(0, 1))
        if self.normalize:
            input = normalize_to_scales(input, scales)


        if self.loss in ['forward-kl', 'backward-kl']:
            # In this case, the parameters consist of means and variances
            means, logvariances = torch.chunk(pred_params, 2, dim=-1)

            #logvariances = torch.clamp(logvariances, min=-10, max=10)

            pred_params = (means, torch.exp(logvariances))


            eval_samples = 25 #TODO in config

            model_predictions = torch.zeros_like(self.eval_model.forward(input, means))
            for i in range(eval_samples):
                param_sample = means + torch.randn_like(logvariances)*torch.exp(logvariances)
                model_predictions += self.eval_model.forward(input, param_sample, scales)
            model_predictions /= float(eval_samples)
        else: model_predictions = self.eval_model.forward(input, pred_params, scales)  # (batch_size, dataset_size, dy)

        if self.normalize: model_predictions = renormalize(model_predictions, scales)
        return model_predictions, pred_params, scales

    def compute_forward(self, batch):
        datasets_in, gt_params, gt_Y = batch
        datasets_in_X = datasets_in[:, :, 0:self.dx]  # the x values for every point in every dataset
        if self.normalize:
            datasets_in_X, scales = normalize_input(datasets_in_X.transpose(0,1))
            datasets_in_X = datasets_in_X.transpose(0,1)
        datasets_in_Y = datasets_in[:, :, self.dx:self.dx + self.dy]  # the y values for every point in every dataset


        # Given a dataset, the transformer predicts some parameters
        # Transpose such that sequence length is first dimension
        pred_params, scales = self(datasets_in.transpose(0, 1))

        if self.normalize and self.loss in ['forward-kl', 'mle-params']:
            gt_params = normalize_params(gt_params, scales)

        if self.loss in ['forward-kl', 'backward-kl']:
            # In this case, the parameters consist of means and variances
            means, logvariances = torch.chunk(pred_params, 2, dim=-1)

            # clamp the logvariances to prevent explosion of loss (do we need this?)
            logvariances = torch.clamp(logvariances, min=-10, max=10)

            if self.loss == 'forward-kl':
                nll = 0.5 * (torch.exp(-logvariances) * (means - gt_params) ** 2 + logvariances + np.log(2 * torch.pi))
                sum = nll.sum(dim=-1)
                #return torch.sum((gt_params-means)**2/torch.exp(logvariances) + logvariances, dim=-1).mean(), datasets_in, datasets_in_Y, pred_params, None
                return sum.mean(), datasets_in, datasets_in_Y, pred_params, None

            # For backward KL, we need to sample parameters using the reparameterization trick to compute the loss
            pred_params = means + torch.randn_like(means) * torch.exp(logvariances) #TODO check sampling correct, chatgpt says a factor of 0.5 is needed

        # Compute the model predictions. For the point estimate, pred_
        model_predictions = self.eval_model.forward(datasets_in_X, pred_params, scales)  # (batch_size, dataset_size, dy)

        if self.normalize:
            model_predictions = renormalize(model_predictions, scales)

        if self.loss in ['backward-kl', 'mle-dataset']:
            mse = torch.mean(torch.sum((datasets_in_Y - model_predictions) ** 2, dim=1), dim=0)

        if self.loss == 'backward-kl': # TODO: add division by noise var!
            return mse + torch.sum(torch.exp(logvariances)-logvariances-1+means**2, dim=-1).mean(), datasets_in, datasets_in_Y, pred_params, model_predictions

        if self.loss == 'mle-dataset':
            # weight decay term
            l2_penalty = weight_decay*torch.sum(pred_params ** 2, dim=-1).mean()
            loss = mse + l2_penalty
            return loss, datasets_in, datasets_in_Y, pred_params, model_predictions

        if self.loss == 'mle-params':
            # simply optimize MSE between predicted parameters and ground truth
            return torch.mean((pred_params - gt_params)**2), datasets_in, datasets_in_Y, pred_params, model_predictions

    def plot_eval(self, eval_data_batch, loss_fns):
        eval_data, gt_params = eval_data_batch

        num_samples = 20
        loss, datasets_in, datasets_in_Y, pred_params, _ = self.compute_forward(eval_data_batch, loss_fns)
        for i in range (1,num_samples):
            loss, datasets_in, datasets_in_Y, x, _ = self.compute_forward(eval_data_batch, loss_fns)
            pred_params = torch.cat((pred_params, x),dim=0)

        X = torch.linspace(-5, 5, 128)

        model_predictions = self.eval_model.forward(X.unsqueeze(0).repeat(num_samples + 1, 1).unsqueeze(-1), torch.cat((pred_params, gt_params), dim=0))
        plt.plot(X.detach().cpu().numpy(), torch.mean(model_predictions[0:num_samples, :, :], dim=0).detach().cpu().numpy())
        plt.plot(X.detach().cpu().numpy(), model_predictions[-1, :, :].detach().cpu().numpy())
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.show()
        plt.savefig("./plot.png")
