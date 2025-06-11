import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(self, dx, dy, dOut, dT, num_heads, num_layers):
        super().__init__(self)
        self.dIn = dx+dy
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
        nn.init.xavier_uniform(self.CLS)

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


class InContextPointEstimator(nn.Module):
    def __init__(self, dx, dy, dOut, dT, num_heads, num_layers):
        super().__init__(self)
        self.transformer = Transformer(dx, dOut, dT, num_heads, num_layers)

    def forward(self, x):
        # Transformer directly maps to parameters
        return self.transformer(x)

    def compute_loss(self, batch, loss_fns):
        X, Y = batch
        self(X)


