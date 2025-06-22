import argparse
import torch
from torch.utils.data import DataLoader
import wandb
from Downscaling_Dataset_Prep import DownscalingDataset
from Experiments import run_experiment
from config_loader import load_config

import xarray as xr
from pathlib import Path
print("Main.py started")

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
        ds = xr.open_dataset(file_path)[[var_name]] 
        datasets.append(ds)

    merged_ds = xr.merge(datasets)
    return merged_ds

def main(config):
    paths = config["data"]
    elevation_path = paths.get("static", {}).get("elevation", None)

    input_train_ds = load_dataset(paths["train"]["input"], config, section="input")
    target_train_ds = load_dataset(paths["train"]["target"], config, section="target")
    train_dataset = DownscalingDataset(input_train_ds, target_train_ds, config, elevation_path=elevation_path)

    input_val_ds = load_dataset(paths["val"]["input"], config, section="input")
    target_val_ds = load_dataset(paths["val"]["target"], config, section="target")
    val_dataset = DownscalingDataset(input_val_ds, target_val_ds, config, elevation_path=elevation_path)

    input_test_ds = load_dataset(paths["test"]["input"], config, section="input")
    target_test_ds = load_dataset(paths["test"]["target"], config, section="target")
    test_dataset = DownscalingDataset(input_test_ds, target_test_ds, config, elevation_path=elevation_path)

    print(f"Using learning rate scheduler: {config['train'].get('scheduler', 'CyclicLR')}")
    print(f"Train samples: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    model, history, final_val_loss = run_experiment(train_dataset, val_dataset, config=config)

    wandb.log({"final_validation_loss": final_val_loss})
    wandb.finish()


if __name__ == "__main__":
    print("Main.py started running")
    config = load_config("config.yaml", ".paths.yaml")
    try:
        main(config)
    except Exception as e:
        import traceback
        print("Exception occurred in Main.py:")
        traceback.print_exc()
    print("Finished running Main")