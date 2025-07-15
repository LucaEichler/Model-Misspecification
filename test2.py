import in_context_models
import datasets
from main import train




dx = 1
dy = 1
dh = 10
dataset_amount = 100
dataset_size = 50
loss = 'backward-kl'
num_iters = 1000000

model_spec = ('NonLinear', {'dh': dh})

model = in_context_models.InContextModel(dx, dy, 32, 4, 5, model_spec[0], loss, **model_spec[1])
dataset = datasets.ContextDataset(dataset_amount, dataset_size, model_spec[0], dx, dy, **model_spec[1])
model_trained = train(model, dataset, iterations=num_iters, batch_size=100,
                  eval_dataset=dataset)

# needs approx. 10k iterations for convergence