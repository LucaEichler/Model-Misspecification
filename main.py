import torch.optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

import plotting
from classical_models import Linear, LinearVariational, NonLinear, NonLinearVariational
import datasets

config_path = ".\config.yaml"

with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.loader.SafeLoader)

    num_datasets = config.get("num_datasets")
    noise_std = config.get("noise_std")


gt_model = NonLinear(dx=1, dy=1, dh=100)
dataset = datasets.PointDataset(1000, gt_model)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

batch_size = 100
iterations = 50000

model = NonLinearVariational(dx=1, dy=1, dh=100)  # check weight initialization

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# weight decay (?)

def train_step(model, optimizer, loss_fns, dataloader, it):
    model.train()
    model.zero_grad()

    batch = next(iter(dataloader))
    loss = model.compute_loss(batch, loss_fns)
    if it % 1000 == 0: plotting.plot_predictions(gt_model, model)
    loss.backward()
    optimizer.step()
    return loss

loss_fns = {"MSE": torch.nn.MSELoss()}
tqdm_batch = tqdm(range(iterations), unit="batch", ncols=100, leave=True)
for it in tqdm_batch:
    loss = train_step(model, optimizer, loss_fns, dataloader, it)
    tqdm_batch.set_postfix({"loss": loss})
