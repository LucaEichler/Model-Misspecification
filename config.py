import torch
import yaml

config_path = "./config.yaml"

with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.loader.SafeLoader)

    num_datasets = config.get("num_datasets")
    noise_std = config.get("noise_std")
    dataset_size_classical = config.get("dataset_size_classical")
    dataset_size_in_context = config.get("dataset_size_in_context")
    dh = config.get("dh")
    dataset_amount = config.get("dataset_amount")

    device = torch.device(config.get("device") if torch.cuda.is_available() else "cpu")
