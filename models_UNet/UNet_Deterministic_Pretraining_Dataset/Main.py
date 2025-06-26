import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import wandb
from Downscaling_Dataset_Prep import DownscalingDataset
from Experiments import run_experiment
from config_loader import load_config

import xarray as xr
from pathlib import Path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

from torch.utils.data import DataLoader
import torch.nn as nn

def evaluate_test(model, test_dataset, config):
    batch_size = config["experiment"].get("batch_size", 32)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)      
            targets = targets.to(device) 
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss}")
    return avg_loss

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
    test_loss = evaluate_test(model, test_dataset, config)
    wandb.log({"test_loss": test_loss})

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