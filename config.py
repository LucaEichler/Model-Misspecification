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
    weight_decay_in_context = config.get("weight_decay_in_context")
    weight_decay_classical = config.get("weight_decay_classical")
    device = torch.device(config.get("device") if torch.cuda.is_available() else "cpu")
    num_iters_classical = config.get("num_iters_classical")
    num_iters_in_context = config.get("num_iters_in_context")
    batch_size_in_context = config.get("batch_size_in_context")

    wandb_project_name = config.get("wandb_project_name")
    wandb_enabled = config.get("wandb_enabled")