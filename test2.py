import in_context_models
import datasets
from classical_models import NonLinear
from main import train, count_parameters
import torch
from config import device
from main import eval_plot



dx = 1
dy = 1
dh = 10
dataset_amount = 10000
dataset_size = 50
loss = 'backward-kl'
num_iters = 10

model_spec = ('NonLinear', {'dh': dh})

model = in_context_models.InContextModel(dx, dy, 256, 4, 4, model_spec[0], loss, **model_spec[1])
print(count_parameters(model))
dataset = datasets.ContextDatasetAlternative(dataset_amount, dataset_size, model_spec[0], dx, dy, batch_size=100, **model_spec[1])
model_trained = train(model, dataset, iterations=num_iters, batch_size=100,
                  eval_dataset=dataset, lr=1e-4)

for i in range(50):
    gt = NonLinear(dx, dy, dh)
    ds = datasets.PointDataset(dataset_size, gt)

    X = torch.linspace(-2,2,25).unsqueeze(1).to(device)

    Y_pred = model_trained.predict(torch.cat((ds.X, ds.Y), dim=-1).unsqueeze(0), X.unsqueeze(0))

    eval_plot("plot", str(i), gt, X, Y_pred)


# needs approx. 10k iterations for convergence