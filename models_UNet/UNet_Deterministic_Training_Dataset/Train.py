
#In present version of Train.py depemding on the type of learning scheduler used

#If CyclicalLR: per batch stepping, 2 times per cycle 
#If ReduceLROnPlateau :  epoch stepping using validation loss depending on whether it gets stuck with no improvement
#Gradient norm logging included to explore vanishing/exploding gradients
#Learning rate also logged per every 20 batches for each epoch.

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm 
import wandb
import json

def train_one_epoch(model, dataloader, optimizer, criterion, scheduler=None, config=None):
    import torch.optim.lr_scheduler as lrs

    model.train()
    running_loss = 0.0
    quick_test = config["experiment"].get("quick_test", False)

    for i, (inputs, targets) in enumerate(tqdm(dataloader, desc="Training")):
        optimizer.zero_grad()
        outputs = model(inputs, targets)
        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient norm (L2) per batch
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm(2).item() ** 2
        grad_norm = total_norm ** 0.5

        optimizer.step()

        if scheduler and not isinstance(scheduler, lrs.ReduceLROnPlateau):
            scheduler.step()

        running_loss += loss.item()

        # Logging per 20 batches for every epocvh
        if i % 20 == 0:
            log_dict = {
                "train_loss_batch": loss.item(),
                "grad_norm": grad_norm
            }
            if scheduler and not isinstance(scheduler, lrs.ReduceLROnPlateau):
                log_dict["lr"] = scheduler.get_last_lr()[0]
            else:
                log_dict["lr"] = optimizer.param_groups[0]["lr"]
            wandb.log(log_dict)

        if quick_test and i == 2:
            break

    return running_loss / (i + 1)


def validate(model, dataloader, criterion, config=None):
    model.eval()
    running_loss = 0.0
    quick_test = config["experiment"].get("quick_test", False)

    with torch.no_grad():
        for j, (inputs, targets) in enumerate(tqdm(dataloader, desc="Validating")):
            outputs = model(inputs, targets)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            if quick_test and j == 2:
                break

    return running_loss / (j + 1)



def checkpoint_save(model, optimizer, epoch, loss, path, inference_path=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)
    print(f"Best model checkpoint saved at: {path}")

    # Save inference weights 
    if inference_path:
        torch.save(model.state_dict(), inference_path)
        print(f"Inference model saved at: {inference_path}")

def save_model_config(config, path):
    import json
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler=None, config=None):
    import torch.optim.lr_scheduler as lrs

    train_cfg = config["train"]
    num_epochs = train_cfg.get("num_epochs", 30)
    checkpoint_path = train_cfg.get("checkpoint_path", "best_model.pth")
    inference_path = train_cfg.get("inference_weights_path", None)
    model_config_path = train_cfg.get("model_config_path", "model_config.json")

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stopping_patience = 5  # Stop after 5 epochs with no improvement

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, config)
        val_loss = validate(model, val_loader, criterion, config)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")

        # Step scheduler
        if scheduler:
            if isinstance(scheduler, lrs.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        # Save best model and inference weights/config
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0  # Reset counter
            checkpoint_save(model, optimizer, epoch+1, val_loss, checkpoint_path, inference_path)
            save_model_config(config, model_config_path)
        else:
            epochs_no_improve += 1
            print(f"No improvement in val loss for {epochs_no_improve} epoch(s).")

        wandb.log({
            "epoch": epoch+1,
            "train_loss_epoch": train_loss,
            "val_loss_epoch": val_loss,
            "lr_epoch": current_lr
        })

        # Early stopping check
        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs with no improvement in val loss for {early_stopping_patience} epochs.")
            break

    return model, history
