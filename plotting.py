import matplotlib.pyplot as plt
import torch

from classical_models import LinearVariational, Linear


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