import argparse
import torch
from torch.utils.data import DataLoader
import wandb
from Downscaling_Dataset_Prep import DownscalingDataset
from Experiments import run_experiment
from config_loader import load_config

import xarray as xr
from pathlib import Path

def load_dataset(file_group: dict, config: dict, section: str) -> xr.Dataset:
    """
    Loading and merging four NetCDF files into one dataset, for features and targets pairs
    
    Args:
        file_group: dict of variable keys -> file paths (from config["data"][split][section])
        config: merged config containing variable names
        section: "input" or "target"
    """
    var_mapping = config["variables"][section]
    datasets = []

    for var_key, var_name in var_mapping.items():
        file_path = Path(file_group[var_key])
        ds = xr.open_dataset(file_path)[[var_name]]  # Select only the needed variable
        datasets.append(ds)

    merged_ds = xr.merge(datasets)
    return merged_ds
def main(config):
    paths = config["data"]
    root = Path(paths["root"])

    # Load and merge training datasets
    input_train_ds = load_dataset(paths["train"]["inputs"], config, section="input")
    target_train_ds = load_dataset(paths["train"]["targets"], config, section="target")
    train_dataset = DownscalingDataset(input_train_ds, target_train_ds, config)

    # Load and merge validation datasets
    input_val_ds = load_dataset(paths["val"]["inputs"], config, section="input")
    target_val_ds = load_dataset(paths["val"]["targets"], config, section="target")
    val_dataset = DownscalingDataset(input_val_ds, target_val_ds, config)

    # Load and merge test datasets
    input_test_ds = load_dataset(paths["test"]["inputs"], config, section="input")
    target_test_ds = load_dataset(paths["test"]["targets"], config, section="target")
    test_dataset = DownscalingDataset(input_test_ds, target_test_ds, config)

    print(f"Using learning rate scheduler: {config['train'].get('scheduler', 'CyclicLR')}")
    print(f"Train samples: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    model, history, final_val_loss = run_experiment(train_dataset, val_dataset, config=config)

    wandb.log({"final_validation_loss": final_val_loss})
    wandb.finish()


