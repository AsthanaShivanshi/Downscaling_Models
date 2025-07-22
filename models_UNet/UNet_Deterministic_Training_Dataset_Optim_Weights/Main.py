import argparse
import torch
from torch.utils.data import DataLoader
import wandb
from Downscaling_Dataset_Prep import DownscalingDataset
from Experiments import run_experiment
from config_loader import load_config
import numpy as np
import xarray as xr
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Main.py started")

def load_dataset(file_group: dict, config: dict, section: str) -> xr.Dataset:
    var_mapping = config["variables"][section]
    datasets = []
    for var_key, var_name in var_mapping.items():
        file_path = Path(file_group[var_key])
        ds = xr.open_dataset(file_path)[[var_name]] 
        datasets.append(ds)
    merged_ds = xr.merge(datasets)
    return merged_ds

def main(config):
    wandb.init(
        project="UNet_Deterministic_Training_Optim_Weights",
        name="single_run",
        config=config,
        reinit=True
    )
    paths = config["data"]
    elevation_path = paths.get("static", {}).get("elevation", None)

    input_train_ds = load_dataset(paths["train"]["input"], config, section="input")
    target_train_ds = load_dataset(paths["train"]["target"], config, section="target")
    train_dataset = DownscalingDataset(input_train_ds, target_train_ds, config, elevation_path=elevation_path)

    input_val_ds = load_dataset(paths["val"]["input"], config, section="input")
    target_val_ds = load_dataset(paths["val"]["target"], config, section="target")
    val_dataset = DownscalingDataset(input_val_ds, target_val_ds, config, elevation_path=elevation_path)

    print(f"Using learning rate scheduler: {config['train'].get('scheduler', 'CyclicLR')}")
    print(f"Train samples: {len(train_dataset)}, Val: {len(val_dataset)}")

    model, history, final_val_loss, best_val_loss, best_val_loss_per_channel = run_experiment(
        train_dataset, val_dataset, config=config
    )

    print({"final val loss per last epoch": final_val_loss, "best val loss across epochs": best_val_loss})

    # Log per-channel best validation losses
    var_names = ["precip", "temp", "tmin", "tmax"]
    for var, loss in zip(var_names, best_val_loss_per_channel):
        print(f"Best validation loss for channel {var}: {loss}")
        wandb.log({f"{var}/val": loss})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick_test", action="store_true", help="Run quick for small amount of samples")
    args = parser.parse_args()

    print("Main.py started running")
    config = load_config("config.yaml", ".paths.yaml")
    if args.quick_test:
        print("Running in quick test mode")
        config["experiment"]["quick_test"] = True

    try:
        main(config)
    except Exception as e:
        import traceback
        print("Exception occurred in Main.py:")
        traceback.print_exc()
    print("Finished running Main")