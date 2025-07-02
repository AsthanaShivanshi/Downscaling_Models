import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import wandb
from Downscaling_Dataset_Prep import DownscalingDataset
from Experiments import run_experiment
from config_loader import load_config
from losses import WeightedMSELoss, WeightedHuberLoss
import numpy as np
import xarray as xr
from pathlib import Path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Main.py started")

def load_dataset(file_group: dict, config: dict, section: str) -> xr.Dataset:
    """
    Loading and merging four NetCDF files into one dataset, for features and targets pairs
    """
    var_mapping = config["variables"][section]
    datasets = []

    for var_key, var_name in var_mapping.items():
        file_path = Path(file_group[var_key])
        ds = xr.open_dataset(file_path)[[var_name]] 
        datasets.append(ds)

    merged_ds = xr.merge(datasets)
    return merged_ds



def evaluate_test(model, test_dataset, config):
    batch_size = config["experiment"].get("batch_size", 32)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    loss_fn_name = config["train"].get("loss_fn", "huber").lower()
    if loss_fn_name == "huber":
        weights = [0.25, 0.25, 0.25, 0.25]
        delta = config["train"].get("huber_delta", 0.05)
        criterion = WeightedHuberLoss(weights=weights, delta=delta)
        def channel_loss_fn(pred, target):
            return [
                nn.functional.huber_loss(pred[:, c], target[:, c], delta=delta, reduction='mean').item()
                for c in range(pred.shape[1])
            ]
    else:
        weights = [0.25, 0.25, 0.25, 0.25]
        criterion = WeightedMSELoss(weights=weights)
        def channel_loss_fn(pred, target):
            return [
                nn.functional.mse_loss(pred[:, c], target[:, c], reduction='mean').item()
                for c in range(pred.shape[1])
            ]


    total_loss = 0.0
    channel_losses_individual = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            # Channel-wise loss
            individual = channel_loss_fn(outputs, targets)
            channel_losses_individual.append(individual)
    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss}")

    # Channel-wise average loss
    var_names = ["precip", "temp", "tmin", "tmax"]
    channel_losses_individual = np.array(channel_losses_individual)
    avg_channel_losses = np.mean(channel_losses_individual, axis=0)
    for var, loss in zip(var_names, avg_channel_losses):
        print(f"Average test loss for channel {var}: {loss}")

    # Log total weighted test loss
    wandb.log({"loss/test": avg_loss})

    # Log per-channel test losses
    for var, loss in zip(var_names, avg_channel_losses):
        wandb.log({f"{var}/test": loss})    

    return avg_loss, avg_channel_losses

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

    model, history, final_val_loss, best_val_loss = run_experiment(train_dataset, val_dataset, config=config)

    print({"final val loss per last epoch": final_val_loss, "best val loss across epochs": best_val_loss})
    test_loss = evaluate_test(model, test_dataset, config)
    print({"test_loss": test_loss})


if __name__ == "__main__":
    import sys
    parser=argparse.ArgumentParser()
    parser.add_argument("--quick_test", action="store_true",help="Run quick for small amount of samples")
    args=parser.parse_args()

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