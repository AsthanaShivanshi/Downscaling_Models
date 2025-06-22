from UNet import UNet
import torch
import torch.nn as nn
from Train import train_model, checkpoint_save
import wandb
from torch.optim.lr_scheduler import CyclicLR
from learning_scheduler import get_scheduler
from torch.optim.lr_scheduler import StepLR

def run_experiment(train_dataset, val_dataset, config):
    train_cfg = config["train"]
    exp_cfg = config["experiment"]

    # Initializing W&B
    wandb.init(
        project=train_cfg.get("wandb_project", "unet_downscaling"),
        name=train_cfg.get("wandb_run_name", "CLR_experiment"),
        config={
            "optimizer": train_cfg.get("optimizer", "Adam"),
            "loss": train_cfg.get("loss_fn", "MSE"),
            "base_lr": train_cfg.get("base_lr", 1e-4),
            "max_lr": train_cfg.get("max_lr", 1e-3),
            "scheduler": train_cfg.get("scheduler", "CyclicLR"),
            "mode": train_cfg.get("scheduler_mode", "triangular"),
            "epochs": train_cfg.get("num_epochs", 20)
        }
    )

    model = UNet(in_channels=train_cfg.get("in_channels", 4), out_channels=train_cfg.get("out_channels", 4))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("max_lr", 1e-3))
    )

    scheduler_name = train_cfg.get("scheduler", "CyclicLR")
    scheduler = get_scheduler(scheduler_name, optimizer, train_cfg)

    criterion = nn.MSELoss()

    wandb.watch(model, log="all", log_freq=100)

    batch_size = exp_cfg.get("batch_size", 32)
    quick_test = exp_cfg.get("quick_test", False)

    if quick_test:
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_dataset, range(100)),
            batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(val_dataset, range(30)),
            batch_size=batch_size, shuffle=False
        )
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    trained_model, history = train_model(
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
        model, optimizer, epoch=train_cfg.get("num_epochs", 20),
        loss=final_val_loss, path=checkpoint_path
    )

    return trained_model, history, final_val_loss

