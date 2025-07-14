import optuna
from config_loader import load_config
from Experiments import run_experiment
from Downscaling_Dataset_Prep import DownscalingDataset
from Main import load_dataset
import matplotlib.pyplot as plt


def objective(trial):
    w0 = trial.suggest_float("w0", 0.01, 1.0)
    w_rest = [trial.suggest_float(f"w{i}", 0.01, 1.0) for i in range(1, 4)]
    weights = [w0] + w_rest
    weights = [w / sum(weights) for w in weights] #Normalising weights adding to 1
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

    config["train"]["num_epochs"] = 10
    _, _, _, best_val_loss, best_val_loss_per_channel = run_experiment(
        train_dataset, val_dataset, config, trial=trial
    )

    trial.set_user_attr("weights", weights)
    trial.set_user_attr("val_loss_per_channel", best_val_loss_per_channel)

    # Return (precip loss, total loss): Aims for val
    return best_val_loss_per_channel[0], best_val_loss

if __name__ == "__main__":
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(objective, n_trials=20)

    print("Best trials (Pareto front):")
    for t in study.best_trials:
        print(f"Trial {t.number}: weights={t.user_attrs['weights']}, per_channel_val_loss={t.user_attrs['val_loss_per_channel']}, values={t.values}")

    # Move plotting OUTSIDE the loop
    pareto_precip = [t.values[0] for t in study.best_trials]
    pareto_total = [t.values[1] for t in study.best_trials]

    plt.figure(figsize=(7,5))
    plt.scatter(pareto_precip, pareto_total, c='red', label='Pareto front')
    plt.xlabel("Precip Channel Loss")
    plt.ylabel("Total Loss")
    plt.title("Pareto Front: Precip vs Total Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("pareto_front.png")# ...existing code...
    plt.tight_layout()
    plt.savefig("pareto_front.png", dpi=150)
    plt.show()

    # Best for precip loss
    best_precip_trial = min(study.best_trials, key=lambda t: t.values[0])
    print("\nBest weights for lowest precip loss:", best_precip_trial.user_attrs["weights"])
    print("Per-channel val loss:", best_precip_trial.user_attrs["val_loss_per_channel"])
    print("Objectives (precip loss, total loss):", best_precip_trial.values)

    # Best for total loss
    best_total_trial = min(study.best_trials, key=lambda t: t.values[1])
    print("\nBest weights for lowest total loss:", best_total_trial.user_attrs["weights"])
    print("Per-channel val loss:", best_total_trial.user_attrs["val_loss_per_channel"])
    print("Objectives (precip loss, total loss):", best_total_trial.values)

    # After Optuna optimization and Pareto plotting
    best_total_trial = min(study.best_trials, key=lambda t: t.values[1])
    best_weights = best_total_trial.user_attrs["weights"]

    print("\nRetraining model with best weights for total loss and saving checkpoint...")
    config = load_config("config.yaml", ".paths.yaml")
    config["train"]["loss_weights"] = best_weights
    config["train"]["num_epochs"] = 100  # or your preferred value

    paths = config["data"]
    elevation_path = paths.get("static", {}).get("elevation", None)
    input_train_ds = load_dataset(paths["train"]["input"], config, section="input")
    target_train_ds = load_dataset(paths["train"]["target"], config, section="target")
    train_dataset = DownscalingDataset(input_train_ds, target_train_ds, config, elevation_path=elevation_path)
    input_val_ds = load_dataset(paths["val"]["input"], config, section="input")
    target_val_ds = load_dataset(paths["val"]["target"], config, section="target")
    val_dataset = DownscalingDataset(input_val_ds, target_val_ds, config, elevation_path=elevation_path)

    from Experiments import run_experiment
    model, history, final_val_loss, best_val_loss, best_val_loss_per_channel = run_experiment(
        train_dataset, val_dataset, config, trial=None
    )
    print("Retraining complete. Best model saved.")