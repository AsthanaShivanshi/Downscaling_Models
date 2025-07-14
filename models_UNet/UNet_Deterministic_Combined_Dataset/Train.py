
#In present version of Train.py depemding on the type of learning scheduler used

#If CyclicalLR: per batch stepping, 2 times per cycle 
#If ReduceLROnPlateau :  epoch stepping using validation loss depending on whether it gets stuck with no improvement
#Gradient norm logging included to explore vanishing/exploding gradients
#Learniugn rate for every epoch : logged
import torch
from tqdm import tqdm 
import wandb
import time
from losses import WeightedHuberLoss, WeightedMSELoss 
import torch.optim.lr_scheduler as lrs
from torch.nn import functional as F


def train_one_epoch(model, dataloader, optimizer, criterion, scheduler=None, config=None):

    model.train()
    running_loss = 0.0
    per_channel_sum = None

    for i, (inputs, targets) in enumerate(tqdm(dataloader, desc="Training")):
        device=next(model.parameters()).device
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, targets)
        loss = criterion(outputs, targets)
        loss.backward()

        # Per-channel loss
        if isinstance(criterion, WeightedHuberLoss):
            delta = criterion.delta
            per_channel = torch.tensor([
                F.huber_loss(outputs[:, c], targets[:, c], delta=delta, reduction='mean').item()
                for c in range(outputs.shape[1])
            ])
        elif isinstance(criterion, WeightedMSELoss):
            per_channel = torch.tensor([
                F.mse_loss(outputs[:, c], targets[:, c], reduction='mean').item()
                for c in range(outputs.shape[1])
            ])
        else:
            per_channel = torch.zeros(outputs.shape[1])

        if per_channel_sum is None:
            per_channel_sum = per_channel
        else:
            per_channel_sum += per_channel

        optimizer.step()

        if scheduler and not isinstance(scheduler, lrs.ReduceLROnPlateau):
            scheduler.step()

        running_loss += loss.item()


    avg_per_channel = (per_channel_sum / (i + 1)).tolist()
    return running_loss / (i + 1), avg_per_channel


def validate(model, dataloader, criterion, config=None):
    model.eval()
    running_loss = 0.0
    per_channel_sum = None

    with torch.no_grad():
        for j, (inputs, targets) in enumerate(tqdm(dataloader, desc="Validating")):
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs, targets)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            # Per-channel loss
            if isinstance(criterion, WeightedHuberLoss):
                delta = criterion.delta
                per_channel = torch.tensor([
                    F.huber_loss(outputs[:, c], targets[:, c], delta=delta, reduction='mean').item()
                    for c in range(outputs.shape[1])
                ])
            elif isinstance(criterion, WeightedMSELoss):
                per_channel = torch.tensor([
                    F.mse_loss(outputs[:, c], targets[:, c], reduction='mean').item()
                    for c in range(outputs.shape[1])
                ])
            else:
                per_channel = torch.zeros(outputs.shape[1])

            if per_channel_sum is None:
                per_channel_sum = per_channel
            else:
                per_channel_sum += per_channel

    avg_per_channel = (per_channel_sum / (j + 1)).tolist()
    return running_loss / (j + 1), avg_per_channel


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

    train_cfg = config["train"]
    num_epochs = train_cfg.get("num_epochs", 100)
    checkpoint_path = train_cfg.get("checkpoint_path", "best_model.pth")
    inference_path = train_cfg.get("inference_weights_path", None)
    model_config_path = train_cfg.get("model_config_path", "model_config.json")

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stopping_patience = train_cfg.get("early_stopping_patience", 3)

    var_names = ["RhiresD", "TabsD", "TminD", "TmaxD"]

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_per_channel = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, config)
        val_loss, val_per_channel = validate(model, val_loader, criterion, config)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Train Loss: {train_loss} | Val Loss: {val_loss}")

        epoch_duration= time.time()-start_time
        print(f"Epoch {epoch+1} duration: {epoch_duration} seconds")
        # Step scheduler
        if scheduler:
            if isinstance(scheduler, lrs.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        # Saving best model and inference weights/config
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_per_channel=val_per_channel
            epochs_no_improve = 0  # Reset counter
            checkpoint_save(model, optimizer, epoch+1, val_loss, checkpoint_path, inference_path)
            save_model_config(config, model_config_path)
        else:
            epochs_no_improve += 1
            print(f"No improvement in val loss for {epochs_no_improve} epoch(s).")

        wandb_log_dict = {
            "epoch": epoch+1,
            "loss/train": train_loss,
            "loss/val": val_loss,
            "lr": current_lr,
            "epoch_time": epoch_duration
        }
        for i, var_name in enumerate(var_names):
            wandb_log_dict[f"{var_name}/train"] = train_per_channel[i]
            wandb_log_dict[f"{var_name}/val"] = val_per_channel[i]

        wandb.log(wandb_log_dict)


        # Early stopping
        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs with no improvement in val loss for {early_stopping_patience} epochs.")
            break
    print(f"best_val_loss: {best_val_loss}")

    return model, history, best_val_loss,best_val_loss_per_channel
