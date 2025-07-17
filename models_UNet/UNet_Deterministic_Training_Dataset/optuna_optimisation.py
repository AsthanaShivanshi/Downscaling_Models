import optuna
from config_loader import load_config
from Experiments import run_experiment
from Downscaling_Dataset_Prep import DownscalingDataset
from Main import load_dataset
import wandb
import numpy as np
import pandas as pd

MAX_VALID_TRIALS = 20

def objective(trial):
    w0 = trial.suggest_float("w0", 0.1, 1.0) #unnormalized weight for precip constrained.
    w_rest = [trial.suggest_float(f"w{i}", 0.1, 1.0) for i in range(1, 4)]
    weights = [w0] + w_rest
    weights = [w / sum(weights) for w in weights]
    # Constraints for other channelés including precip (normalised)
    if weights[0] < 0.25 or any(w < 0.10 for w in weights[1:]):
        raise optuna.TrialPruned()
    print(f"Trial {trial.number}: Normalized weights used: {weights}, sum={sum(weights)}")

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
    _, _, _, best_val_loss, best_val_loss_per_channel = run_experiment(
        train_dataset, val_dataset, config, trial=trial
    )

    trial.set_user_attr("weights", weights)
    trial.set_user_attr("val_loss_per_channel", best_val_loss_per_channel)

    # Log to wandb for each valid trial
    wandb.log({
        "trial": trial.number,
        "weights": weights,
        "total_val_loss": best_val_loss,
        "precip_val_loss": best_val_loss_per_channel[0],
        "temp_val_loss": best_val_loss_per_channel[1],
        "tmin_val_loss": best_val_loss_per_channel[2],
        "tmax_val_loss": best_val_loss_per_channel[3],
    })

    # precip loss, total loss
    return best_val_loss_per_channel[0], best_val_loss

if __name__ == "__main__":
    study = optuna.create_study(directions=["minimize", "minimize"])
    valid_trials = 0
    trial_data = []

    for _ in range(500):  # Large upper bound to make uper limit flkexible
        if valid_trials >= MAX_VALID_TRIALS:
            print(f"Reached {MAX_VALID_TRIALS} valid trials. Stopping optimisation.")
            break
        trial = study.ask()
        try:
            values = objective(trial)
            study.tell(trial, values)
            if trial.state == optuna.trial.TrialState.COMPLETE:
                valid_trials += 1
                trial_data.append({
                    "trial": trial.number,
                    "weights": trial.user_attrs.get("weights"),
                    "precip_loss": values[0],
                    "total_loss": values[1],
                    "per_channel_val_loss": trial.user_attrs.get("val_loss_per_channel")
                })
        except optuna.TrialPruned:
            study.tell(trial, None, state=optuna.trial.TrialState.PRUNED)
            continue

    df = pd.DataFrame(trial_data)
    print("\nAll valid trials:")
    print(df)
    df.to_csv("optuna_trials_table.csv", index=False)