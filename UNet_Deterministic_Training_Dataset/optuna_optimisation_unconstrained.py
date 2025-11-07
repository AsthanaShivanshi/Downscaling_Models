import optuna
from config_loader import load_config
from Experiments import run_experiment
from Downscaling_Dataset_Prep import DownscalingDataset
from Main import load_dataset
import wandb
import numpy as np
import pandas as pd
import json

MAX_VALID_TRIALS = 40

def objective(trial):
    wandb.init(
        project="UNet_Deterministic",
        name=f"UNCONSTRAINED_trial_third_iter_{trial.number}",
        config={},
        reinit=True
    )
    # Suggest unnormalized weights
    initial_weights = [
        trial.suggest_float("precip_weight", 0.1, 1.0),
        trial.suggest_float("temp_weight", 0.1, 1.0),
        trial.suggest_float("tmin_weight", 0.1, 1.0),
        trial.suggest_float("tmax_weight", 0.1, 1.0)
    ]

    # Normalizing and taking all trials : unconstrained.: no channel wise constraints
    weights = [w / sum(initial_weights) for w in initial_weights]
    print("initial unnormalised weights:", initial_weights)
    print(f"Trial {trial.number}: Normalized weights used: {weights}, sum={sum(weights)}")
    trial.set_user_attr("normalized_weights", weights)
    trial.set_user_attr("weights", initial_weights)
    config = load_config("config.yaml", ".paths.yaml")
    config["train"]["loss_weights"] = weights 

    paths = config["data"]
    elevation_path = paths.get("static", {}).get("elevation", None)
    input_train_ds = load_dataset(paths["train"]["input"], config, section="input")
    target_train_ds = load_dataset(paths["train"]["target"], config, section="target")
    train_dataset = DownscalingDataset(input_train_ds, target_train_ds, config, elevation_path=elevation_path)
    input_val_ds = load_dataset(paths["val"]["input"], config, section="input")
    target_val_ds = load_dataset(paths["val"]["target"], config, section="target")
    val_dataset = DownscalingDataset(input_val_ds, target_val_ds, config, elevation_path=elevation_path)

    config["train"]["num_epochs"] = 20
    model, history, final_val_loss, best_val_loss, best_val_loss_per_channel = run_experiment(
        train_dataset, val_dataset, config, trial=trial
    )

    trial.set_user_attr("val_loss_per_channel", best_val_loss_per_channel)
    trial.set_user_attr("epoch_history", history)

    wandb.log({
        "trial": trial.number,
        "initial_weights": initial_weights,
        "weights": weights,
        "total_val_loss": best_val_loss,
        "precip_val_loss": best_val_loss_per_channel[0],
        "temp_val_loss": best_val_loss_per_channel[1],
        "tmin_val_loss": best_val_loss_per_channel[2],
        "tmax_val_loss": best_val_loss_per_channel[3],
    })
    wandb.finish()
    return best_val_loss_per_channel[0], best_val_loss

if __name__ == "__main__":
    study = optuna.create_study(directions=["minimize", "minimize"])
    valid_trials = 0
    trial_data = []

    existing_trials = len(study.trials)
    start_trial = 1

    for trial_idx in range(start_trial, 1000):
        if valid_trials >= MAX_VALID_TRIALS:
            print(f"Reached {MAX_VALID_TRIALS} valid trials. Stopping optimisation.")
            break
        trial = study.ask()
        try:
            values = objective(trial)
            study.tell(trial, values)
            last_trial = study.trials[-1]
            if last_trial.state == optuna.trial.TrialState.COMPLETE:
                valid_trials += 1
                trial_data.append({
                    "trial": last_trial.number,
                    "initial_weights": last_trial.user_attrs.get("weights"),
                    "normalized_weights": last_trial.user_attrs.get("normalized_weights"),
                    "precip_loss": values[0],
                    "total_loss": values[1],
                    "per_channel_val_loss": last_trial.user_attrs.get("val_loss_per_channel"),
                    "epoch_history": last_trial.user_attrs.get("epoch_history")
                })
        except optuna.TrialPruned:
            study.tell(trial, None, state=optuna.trial.TrialState.PRUNED)
            continue