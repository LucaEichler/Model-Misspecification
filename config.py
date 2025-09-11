import torch
import yaml

config_path = "./config.yaml"

with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.loader.SafeLoader)

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
    test_trials = config.get("test_trials")

    wandb_project_name = config.get("wandb_project_name")
    wandb_enabled = config.get("wandb_enabled")
    wandb_exp_name = config.get("wandb_exp_name")

    lr_in_context = config.get("lr_in_context")
    lr_classical = config.get("lr_classical")

    lambda_mle = config.get("lambda_mle")

    early_stopping_enabled = config.get("early_stopping_enabled")
    early_stopping_patience = config.get("early_stopping_patience")
    early_stopping_delta = config.get("early_stopping_delta")

    early_stopping_params = {
        'early_stopping_enabled': early_stopping_enabled,
        'patience': early_stopping_patience,
        'min_delta': early_stopping_delta
    }

    load_best_model = config.get("load_best_model")