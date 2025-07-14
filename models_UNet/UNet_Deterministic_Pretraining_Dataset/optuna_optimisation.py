import optuna
from config_loader import load_config
from Experiments import run_experiment
from Downscaling_Dataset_Prep import DownscalingDataset
from Main import load_dataset

def objective(trial):
    # Precip weight: at least 0.25 artificial constraint
    w0 = trial.suggest_float("w0", 0.25, 1.0)
    w_rest = [trial.suggest_float(f"w{i}", 0.01, 1.0) for i in range(1, 4)]
    weights = [w0] + w_rest
    # Normalize so weights sum to 1
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

    config["train"]["num_epochs"] = 10
    _, _, _, best_val_loss = run_experiment(train_dataset, val_dataset, config,  
                                            trial=trial)
    return best_val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    print("Best weights:", study.best_params)
    print("Best validation loss:", study.best_value)