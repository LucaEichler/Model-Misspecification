from torch.utils.data import DataLoader
import pandas as pd

import datasets

from classical_models import Linear, NonLinear, LinearVariational
from main import train

# Check how many points are needed for classical models to converge to the ground truth
# Where possible, compare with closed form solution


dx = 1
dy = 1
dh = 10
num_iters = 10000
order = 1

def get_model_from_name(name):
    if model_name == "linear":
        model = Linear(2, dim, order=1)
    if model_name == "linear-2":
        model = Linear(2, dim, order=2)
    if model_name == "nonlinear":
        model = NonLinear(2, dim, dh=dh)
    return model

results = []
tries = 10
for dim in [1, 10, 100]:
    for noise in [0., 0.01, 0.05, 0.1]:
            for dataset_size in range(1, 51):
                mse_sum = 0
                for _i in range(0,tries):
                    for model_name in ["linear", "linear-2", "nonlinear"]:
                        gt = get_model_from_name(model_name)
                        ds = datasets.PointDataset(dataset_size, gt, noise_std=noise)
                        model = get_model_from_name(model_name)
                        model_trained = train(model, ds, iterations=num_iters, batch_size=min(dataset_size, 100), gt_model=gt, plot=False)


                        # calculate closed form MLE solution
                        X, Y = next(iter(DataLoader(ds, batch_size=dataset_size, shuffle=True)))  # get all data points
                        #closed_form_mle_parameters = model_trained.closed_form_solution(X, Y)

                        mse = ((gt.get_W() - model_trained.get_W())**2).mean()
                        mse_sum += mse
                        """print("Difference gt parameters - trained params: "+str(mse))"""
                        """print("gt parameters: "+str(gt.get_W()))
                        print("trained parameters: "+str(model_trained.get_W()))
                        print("closed form mle parameters: "+str(closed_form_mle_parameters))"""
                mse_sum /= tries
                results.append({'model': model_name, 'dataset_size': dataset_size, 'dimensionality': dim, 'noise': noise, 'mse': mse_sum.item()})
                df = pd.DataFrame(results)

                # Save to disk (choose one or both)
                df.to_csv("test1_results.csv", index=False)

import seaborn as sns
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("test1_results.csv")

# Example plot: metric vs. noise, grouped by dimensionality
sns.lineplot(data=df, x='noise', y='mse', hue='dimensionality')
plt.title("Ablations")
plt.show()