import optuna
from config_loader import load_config
from Experiments import run_experiment
from Downscaling_Dataset_Prep import DownscalingDataset
from Main import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def objective(trial):
    w0 = trial.suggest_float("w0", 0.1, 1.0) 
    w_rest = [trial.suggest_float(f"w{i}", 0.1, 1.0) for i in range(1, 4)]
    weights = [w0] + w_rest
    weights = [w / sum(weights) for w in weights]
    # Constraints
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

    # precip loss, total loss
    return best_val_loss_per_channel[0], best_val_loss

def pareto_front(df):
    df_sorted = df.sort_values("precip_loss")
    pareto = []
    min_total = float("inf")
    for _, row in df_sorted.iterrows():
        if row["total_loss"] < min_total:
            pareto.append(row)
            min_total = row["total_loss"]
    return pd.DataFrame(pareto)

def find_elbow(pareto_df):
    # Get endpoints
    p1 = pareto_df[["precip_loss", "total_loss"]].iloc[0].values
    p2 = pareto_df[["precip_loss", "total_loss"]].iloc[-1].values
    def point_line_dist(point, p1, p2):
        return np.abs(np.cross(p2-p1, p1-point)) / np.linalg.norm(p2-p1)
    distances = pareto_df.apply(
        lambda row: point_line_dist(row[["precip_loss", "total_loss"]].values, p1, p2), axis=1
    )
    elbow_idx = distances.idxmax()
    return pareto_df.loc[elbow_idx]

if __name__ == "__main__":
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(objective, n_trials=30)

    # All valid trials
    trial_data = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            trial_data.append({
                "trial": t.number,
                "weights": t.user_attrs.get("weights"),
                "precip_loss": t.values[0] if t.values else None,
                "total_loss": t.values[1] if t.values else None,
                "per_channel_val_loss": t.user_attrs.get("val_loss_per_channel")
            })
    df = pd.DataFrame(trial_data)

    print("All valid trials:")
    print(df)

    pareto_df = pareto_front(df)
    elbow_trial = find_elbow(pareto_df)
    best_tradeoff_trial = study.trials[elbow_trial["trial"]]

    plt.figure(figsize=(8,6))
    plt.scatter(df["precip_loss"], df["total_loss"], c='gray', label='All valid trials')
    plt.plot(pareto_df["precip_loss"], pareto_df["total_loss"], c='red', label='Pareto front')
    plt.scatter(
        elbow_trial["precip_loss"],
        elbow_trial["total_loss"],
        c='blue', s=120, marker='*', label=f'Elbow (Trial {elbow_trial["trial"]})'
    )
    plt.annotate(
        f'Trial {elbow_trial["trial"]}\n({elbow_trial["precip_loss"]:.2e}, {elbow_trial["total_loss"]:.2e})',
        (elbow_trial["precip_loss"], elbow_trial["total_loss"]),
        textcoords="offset points", xytext=(30,-30), ha='center', fontsize=12, color="blue",
        arrowprops=dict(arrowstyle="->", color="blue")
    )
    plt.xlabel("Precip Channel Loss")
    plt.ylabel("Total Loss")
    plt.title("Pareto Front: Precip vs Total Loss (All Valid Trials)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pareto_front_all.png", dpi=500)
    plt.show()

    print("\nRetraining model with best tradeoff weights and saving checkpoint...")
    config = load_config("config.yaml", ".paths.yaml")
    config["train"]["loss_weights"] = best_tradeoff_trial.user_attrs["weights"]
    config["train"]["num_epochs"] = 100

    paths = config["data"]
    elevation_path = paths.get("static", {}).get("elevation", None)
    input_train_ds = load_dataset(paths["train"]["input"], config, section="input")
    target_train_ds = load_dataset(paths["train"]["target"], config, section="target")
    train_dataset = DownscalingDataset(input_train_ds, target_train_ds, config, elevation_path=elevation_path)
    input_val_ds = load_dataset(paths["val"]["input"], config, section="input")
    target_val_ds = load_dataset(paths["val"]["target"], config, section="target")
    val_dataset = DownscalingDataset(input_val_ds, target_val_ds, config, elevation_path=elevation_path)

    model, history, final_val_loss, best_val_loss, best_val_loss_per_channel = run_experiment(
        train_dataset, val_dataset, config, trial=None
    )
    print(f"Best model checkpoint saved at: {config['train'].get('checkpoint_path', 'best_model.pth')}")
    print("Retraining complete. Best model (trade-off) saved.")