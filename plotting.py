import matplotlib.pyplot as plt
import torch
from plotly.subplots import make_subplots

import datasets
from classical_models import LinearVariational, Linear
import plotly.graph_objects as go

from main import train


def normalize_minus1_1(x):
    # Compute min and max
    x_min = x.min()
    x_max = x.max()

    # Avoid division by zero
    scale = x_max - x_min
    if scale == 0:
        return torch.zeros_like(x)

    # Normalize to [0,1]
    x_norm = (x - x_min) / scale

    # Rescale to [-1,1]
    x_norm = x_norm * 2 - 1
    return x_norm

def plot_3d_surfaces(model1, model2, W1, W2, model_name="model", ds_name="ds"):
    surfaces = []
    surfaces2 = []
    N = 10


    x_min, y_min = -10., -10.
    x_max, y_max = 10., 10.
    x_points, y_points = 20, 20
    x = torch.linspace(x_min, x_max, x_points)
    y = torch.linspace(y_min, y_max, y_points)
    X1, X2 = torch.meshgrid(x, y, indexing='ij')

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'surface'}, {'type': 'surface'}]])

    for k in range(N):

        x3 = 1.*k

        points = torch.stack([X1.reshape(-1), X2.reshape(-1), x3*torch.ones_like(X1).reshape(-1)], dim=-1)


        if W1 is not None:
            Y = model1(points.unsqueeze(0), W=W1.unsqueeze(0))
        else:
            Y = model1(points)
        if W2 is not None:
            Y2 = model2(points.unsqueeze(0), W=W2.unsqueeze(0))
        else:
            Y2 = model2(points)

        sfc = normalize_minus1_1(Y.reshape(20, 20))
        z1 = sfc.detach().cpu().numpy()
        sfc2 = normalize_minus1_1(Y2.reshape(20, 20))
        z2 = sfc2.detach().cpu().numpy()


        surfaces.append(go.Surface(z=z1, showscale=False, opacity=0.9))
        surfaces2.append(go.Surface(z=z2, showscale=False, opacity=0.9))
    fig.add_trace(surfaces[0], row=1, col=1)
    fig.add_trace(surfaces2[0], row=1, col=2)
    frames = [go.Frame(data=[surfaces[k], surfaces2[k]]) for k in range(N)]
    fig.frames = frames
    fig.update_layout(updatemenus=[dict(type="buttons",buttons=[dict(label="Play",method="animate",args=[None, {"frame": {"duration": 200, "redraw": True},"fromcurrent": True, "transition": {"duration": 0}}])])])
    fig.write_html("./plots/"+model_name+" - "+ds_name+".html")


import config

#model2 = Linear(dx=3, dy=1, order=3, nonlinear_features_enabled=True, feature_sampling_enabled=False)
#model2 = train(model2, ds, valset=ds_val, valfreq=1000, iterations=num_iters, batch_size=100, lr=config.lr_classical, use_wandb=config.wandb_enabled)

#plot_3d_surfaces(model2, model2, None, W2=model2.closed_form_solution_regularized(ds.X, ds.Y, lambd=config.lambda_mle))

def plot_predictions(gt_model, model):
    """
    Plot the predictions of a training model against the ground truth.
    """
    #model2 = Linear(dx=1 ,dy=1, order=2)
    #model2.W = torch.nn.Parameter(model.W)

    X = torch.linspace(0, 1, 128*model.dx).view(128, 1*model.dx)
    Y_gt = gt_model(X)
    Y = model(X, plot=True)
    plt.plot(X[:, 0].detach().numpy(), Y.detach().numpy())
    plt.plot(X[:, 0].detach().numpy(), Y_gt.detach().numpy())
    plt.show()