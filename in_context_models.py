import torch
from matplotlib import pyplot as plt
from torch import nn
from classical_models import Linear


class Transformer(nn.Module):
    def __init__(self, dx, dy, dOut, dT, num_heads, num_layers):
        super().__init__()
        self.dIn = dx + dy
        self.dOut = dOut
        self.dT = dT

        # This layer maps the input of dim dx+dy to the desired Transformer dimensionality dT
        self.encoder = nn.Linear(self.dIn, dT)

        # This layer maps from the Transformer output of dim dT to the desired output dim dT
        self.decoder = nn.Linear(dT, dOut)

        # init the actual Transformer
        tsf_layer = nn.TransformerEncoderLayer(d_model=dT, nhead=num_heads, dim_feedforward=4 * dT, batch_first=False)
        self.transformer_model = nn.TransformerEncoder(tsf_layer, num_layers=num_layers)

        # The CLS token which will contain the result after the transformer is applied
        self.CLS = nn.Parameter(torch.zeros(1, 1, dT))
        nn.init.xavier_uniform_(self.CLS)

    def forward(self, x):
        # x is of shape (seq_length, batch_size, dx)
        x_emb = self.encoder(x)

        # repeat the CLS token to match the batch size
        cls = self.CLS.repeat(1, x.shape[1], 1)

        # concatenate (prepend) CLS to the input batch
        tf_in = torch.cat((cls, x_emb), dim=0)

        # forward through transformer
        tf_out = self.transformer_model(tf_in)

        # map to desired output dimension (only take output at CLS token) & return
        return self.decoder(tf_out[0, :, :])


class InContextModel(nn.Module):
    def __init__(self, dx, dy, dT, num_heads, num_layers, output_model, loss, **kwargs):
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.loss = loss

        # Create model which will be used for evaluation and freeze its parameters
        # With the batched forward, freezing should not be needed, maybe remove later
        self.eval_model = eval(output_model)(dx, dy, kwargs['order'] if 'order' in kwargs else kwargs['dh'])
        for param in self.eval_model.parameters():
            param.requires_grad = False

        # output of transformer should be of the size of the parameter vector for a model
        # (or mean and variance for bayesian methods i.e. *2)
        dOut = self.eval_model.count_params()
        if self.loss in ['forward-kl', 'backward-kl']:
            dOut *=2

        self.transformer = Transformer(dx, dy, dOut, dT, num_heads, num_layers)

    def forward(self, x):
        # Transformer directly maps to parameters
        return self.transformer(x)

    def compute_loss(self, batch, loss_fns):
        datasets_in, gt_params = batch
        datasets_in_X = datasets_in[:, :, 0:self.dx]  # the x values for every point in every dataset
        datasets_in_Y = datasets_in[:, :, self.dy:self.dx + self.dy]  # the y values for every point in every dataset

        # Given a dataset, the transformer predicts some parameters
        # Transpose such that sequence length is first dimension
        pred_params = self(datasets_in.transpose(0, 1))

        if self.loss in ['forward-kl', 'backward-kl']:
            # In this case, the parameters consist of means and variances
            # We need to sample parameters to compute the loss
            means, logvariances = torch.chunk(pred_params, 2, dim=-1)

            # clamp the logvariances to prevent explosion of loss (do we need this?)
            logvariances = torch.clamp(logvariances, min=-10, max=10)

            pred_params = means + torch.randn_like(means) * torch.exp(logvariances)

            if self.loss == 'forward-kl':
                return torch.sum((pred_params-means)**2/torch.exp(logvariances) + logvariances, dim=-1).mean(), datasets_in, datasets_in_Y, pred_params, None

        model_predictions = self.eval_model.forward(datasets_in_X, pred_params)  # (batch_size, dataset_size, dy)

        mse = torch.mean(torch.sum((datasets_in_Y - model_predictions) ** 2, dim=1), dim=0)

        if self.loss == 'backward-kl':
            return mse + torch.sum(torch.exp(logvariances)-logvariances-1+means**2, dim=-1).mean(), datasets_in, datasets_in_Y, pred_params, model_predictions

        if self.loss == 'mle-dataset':
            return mse, datasets_in, datasets_in_Y, pred_params, model_predictions

        if self.loss == 'mle-params':
            # simply optimize MSE between predicted parameters and ground truth
            return loss_fns["MSE"](pred_params, gt_params), datasets_in, datasets_in_Y, pred_params, model_predictions

    def plot_eval(self, eval_data_batch, loss_fns):
        eval_data, gt_params = eval_data_batch
        loss, datasets_in, datasets_in_Y, pred_params, _ = self.compute_loss(eval_data_batch, loss_fns)

        X = torch.linspace(0, 1, 128)

        model_predictions = self.eval_model.forward(torch.stack((X, X), dim=0).unsqueeze(-1), torch.stack((pred_params, gt_params), dim=0))
        plt.plot(X.detach().numpy(), model_predictions[0, :, :].detach().numpy())
        plt.plot(X.detach().numpy(), model_predictions[1, :, :].detach().numpy())
        plt.show()