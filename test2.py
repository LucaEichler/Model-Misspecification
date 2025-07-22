import in_context_models
import datasets
from classical_models import NonLinear
from main import train
import torch
from config import device
from main import eval_plot



dx = 1
dy = 1
dh = 10
dataset_amount = 10000
dataset_size = 50
loss = 'forward-kl'
num_iters = 10000

model_spec = ('NonLinear', {'dh': dh})

model = in_context_models.InContextModel(dx, dy, 32, 4, 5, model_spec[0], loss, **model_spec[1])
dataset = datasets.ContextDataset(dataset_amount, dataset_size, model_spec[0], dx, dy, **model_spec[1])
model_trained = train(model, dataset, iterations=num_iters, batch_size=100,
                  eval_dataset=dataset, lr=0.000001)

for i in range(10):
    gt = NonLinear(dx, dy, dh)
    ds = datasets.PointDataset(dataset_size, gt)

    X = torch.randn(128).unsqueeze(1).to(device)

    Y_pred = model_trained.predict(torch.cat((ds.X, ds.Y), dim=-1).unsqueeze(0), X.unsqueeze(0))

    eval_plot("plot", str(i), gt, X, Y_pred)


# needs approx. 10k iterations for convergence