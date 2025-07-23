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

def run_experiment(train_dataset, val_dataset, config, model=None):
    train_cfg = config["train"]
    exp_cfg = config["experiment"]

    weights = train_cfg.get("loss_weights", [0.25, 0.25, 0.25, 0.25])
    for w in weights:
        if not (0.1 <= w <= 1.0):
            raise ValueError(f"Weight {w} is out of allowed range [0.1, 1.0]")

    # --- CHANGE: Only create a new model if not provided ---
    if model is None:
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
        config=config
    )

    final_val_loss = history['val_loss'][-1]

    checkpoint_path = train_cfg.get("checkpoint_path", "best_model.pth")
    checkpoint_save(
        model, optimizer, epoch=train_cfg.get("num_epochs", 200),
        loss=final_val_loss, path=checkpoint_path
    )

    wandb.log({"best_val_loss": best_val_loss})
    wandb.log({"best_val_loss_per_channel": best_val_per_channel})
    wandb.log({"final_val_loss": final_val_loss})
    return trained_model, history, final_val_loss, best_val_loss, best_val_per_channel