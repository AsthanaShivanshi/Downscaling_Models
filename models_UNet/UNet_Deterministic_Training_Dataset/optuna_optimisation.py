import optuna
from config_loader import load_config
from Experiments import run_experiment
from Downscaling_Dataset_Prep import DownscalingDataset
from Main import load_dataset

def objective(trial):
    w0 = trial.suggest_float("w0", 0.01, 1.0)
    w_rest = [trial.suggest_float(f"w{i}", 0.01, 1.0) for i in range(1, 4)]
    weights = [w0] + w_rest
    weights = [w / sum(weights) for w in weights]

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

    config["train"]["num_epochs"] = 15
    _, _, _, best_val_loss, best_val_loss_per_channel = run_experiment(
        train_dataset, val_dataset, config, trial=trial
    )

    trial.set_user_attr("weights", weights)
    trial.set_user_attr("val_loss_per_channel", best_val_loss_per_channel)

    # Return (precip loss, total loss): Aims for val
    return best_val_loss_per_channel[0], best_val_loss

if __name__ == "__main__":
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(objective, n_trials=10)

    print("Best trials (Pareto front):")
    for t in study.best_trials:
        print(f"Trial {t.number}: weights={t.user_attrs['weights']}, per_channel_val_loss={t.user_attrs['val_loss_per_channel']}, values={t.values}")