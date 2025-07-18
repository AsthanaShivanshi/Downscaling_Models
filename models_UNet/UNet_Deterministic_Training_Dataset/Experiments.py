from UNet import UNet
import torch
import torch.nn as nn
from Train import train_model, checkpoint_save
import wandb
from torch.optim.lr_scheduler import CyclicLR
from learning_scheduler import get_scheduler
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from losses import WeightedHuberLoss, WeightedMSELoss

def run_experiment(train_dataset, val_dataset, config, trial=None):
    train_cfg = config["train"]
    exp_cfg = config["experiment"]

    # W&B logging: Start a new run for each trial, set config at init, use reinit=True
    if trial is not None:
        wandb_run = wandb.init(
            project=train_cfg.get("wandb_project", "unet_downscaling"),
            name=f"{train_cfg.get('wandb_run_name', 'CLR_experiment')}_trial_{trial.number}",
            config={
                "optimizer": train_cfg.get("optimizer", "Adam"),
                "loss": train_cfg.get("loss_fn", "MSE"),
                "base_lr": train_cfg.get("base_lr", 1e-4),
                "max_lr": train_cfg.get("max_lr", 1e-3),
                "scheduler": train_cfg.get("scheduler", "CyclicLR"),
                "mode": train_cfg.get("scheduler_mode", "triangular"),
                "epochs": train_cfg.get("num_epochs", 100),
                "optuna_trial": trial.number,
                "loss_weights": train_cfg.get("loss_weights", [0.25, 0.25, 0.25, 0.25])
            },
            reinit=True
        )
    else:
        wandb_run = wandb.init(
            project=train_cfg.get("wandb_project", "unet_downscaling"),
            name=train_cfg.get("wandb_run_name", "CLR_experiment"),
            config={
                "optimizer": train_cfg.get("optimizer", "Adam"),
                "loss": train_cfg.get("loss_fn", "MSE"),
                "base_lr": train_cfg.get("base_lr", 1e-4),
                "max_lr": train_cfg.get("max_lr", 1e-3),
                "scheduler": train_cfg.get("scheduler", "CyclicLR"),
                "mode": train_cfg.get("scheduler_mode", "triangular"),
                "epochs": train_cfg.get("num_epochs", 100),
                "loss_weights": train_cfg.get("loss_weights", [0.25, 0.25, 0.25, 0.25])
            },
            reinit=True
        )

    model = UNet(in_channels=train_cfg.get("in_channels", 5), out_channels=train_cfg.get("out_channels", 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("max_lr", 1e-3))
    )

    scheduler_name = train_cfg.get("scheduler", "CyclicLR")
    scheduler = get_scheduler(scheduler_name, optimizer, train_cfg)
        
    loss_fn_name = train_cfg.get("loss_fn", "huber").lower()
    weights = train_cfg.get("loss_weights", [0.25, 0.25, 0.25, 0.25])
    if loss_fn_name == "huber":
        criterion = WeightedHuberLoss(weights=weights, delta=train_cfg.get("huber_delta", 0.05))
    elif loss_fn_name == "mse":
        criterion = WeightedMSELoss(weights=weights)
    else:
        raise ValueError(f"Unknown loss function: {loss_fn_name}")
    criterion.to(device)

    batch_size = exp_cfg.get("batch_size", 32)
    quick_test = exp_cfg.get("quick_test", False)

    if quick_test:
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_dataset, range(1000)),
            batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(val_dataset, range(300)),
            batch_size=batch_size, shuffle=False
        )
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    trained_model, history, best_val_loss, best_val_per_channel = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler=scheduler,
        config=config,
        trial=trial
    )

    final_val_loss = history['val_loss'][-1]

    # Not saving checkpoints during trial and optim phase.
    if trial is None:
        checkpoint_path = train_cfg.get("checkpoint_path", "best_model.pth")
        checkpoint_save(
            model, optimizer, epoch=train_cfg.get("num_epochs", 100),
            loss=final_val_loss, path=checkpoint_path
        )

    wandb.log({"best_val_loss": best_val_loss})
    wandb.log({"best_val_loss_per_channel": best_val_per_channel})

    wandb.finish()

    return trained_model, history, final_val_loss, best_val_loss, best_val_per_channel