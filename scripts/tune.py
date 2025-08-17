import optuna
import yaml
import torch
import copy
from datetime import datetime
from cnn_occlusion_robustness.train import run_training_trial

# --- Load the base configuration once ---
try:
    with open("configs/eval/matrix.yaml", "r") as f:
        base_config = yaml.safe_load(f)
except FileNotFoundError:
    print(
        "Error: `configs/eval/matrix.yaml` not found. Make sure you are running this script from the project's root directory."
    )
    exit()


def objective(trial: optuna.trial.Trial) -> float:
    """
    An expanded objective function that also tunes the number of conv layers.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trial_config = copy.deepcopy(base_config)

    # --- Standard Hyperparameters ---
    trial_config["learning_rate"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    # --- Architectural Hyperparameters ---

    # 1. Suggest the number of convolutional layers
    n_conv_layers = trial.suggest_int("n_conv_layers", 1, 2)

    # 2. Suggest kernel size for all conv layers
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5])

    # 3. Dynamically build the architecture list
    new_architecture = []
    in_channels = 3  # Initial input channels for the first layer

    for i in range(n_conv_layers):
        # Suggest the number of filters for this specific layer
        out_channels = trial.suggest_categorical(
            f"conv_{i+1}_out_channels", [8, 16, 32]
        )

        new_architecture.extend(
            [
                {
                    "type": "Conv2d",
                    "params": {
                        "in_channels": in_channels,
                        "out_channels": out_channels,
                        "kernel_size": kernel_size,
                    },
                },
                {"type": "ReLU"},
                {"type": "MaxPool2d", "params": {"kernel_size": 2, "stride": 2}},
            ]
        )
        in_channels = out_channels  # The output of this layer is the input for the next

    # Add the final dense layers
    new_architecture.extend(
        [
            {"type": "Flatten"},
            {
                "type": "Linear",
                "params": {
                    "out_features": trial.suggest_categorical(
                        "fc1_out_features", [64, 84, 120]
                    )
                },
            },
            {"type": "ReLU"},
            {
                "type": "Dropout",
                "params": {"p": trial.suggest_float("dropout_p", 0.1, 0.5)},
            },
            {"type": "Linear", "params": {"out_features": 43}},  # Final output layer
        ]
    )

    # Replace the original architecture with our dynamically generated one
    trial_config["model"]["architecture"] = new_architecture

    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"  Params: {trial.params}")

    try:
        best_accuracy = run_training_trial(
            config=trial_config,
            save_path="/dev/null",
            train_effect_name="none",
            device=device,
            trial=trial,
        )
    except optuna.TrialPruned:
        print(f"--- Trial {trial.number} pruned! ---")
        raise
    except Exception as e:
        print(f"--- Trial {trial.number} failed with an error: {e} ---")
        raise (e)

    return best_accuracy


# --- Main execution block ---
if __name__ == "__main__":
    # 1. Create a pruner for aggressive early-stopping
    #    ASHA is a modern, efficient pruner.
    pruner = optuna.pruners.SuccessiveHalvingPruner()

    # 2. Create a study
    #    We use a database for storage, which lets you pause and resume the study.
    study_name = f"gtsrb-tuning-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    storage_name = "sqlite:///hpo_studies.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",  # We want to maximize validation accuracy
        pruner=pruner,
    )

    print(f"🚀 Starting hyperparameter tuning study: {study_name}")
    print(f"Results will be saved to: {storage_name}")

    try:
        # 3. Start the optimization
        #    `n_jobs=-1` uses all available CPU cores to run trials in parallel.
        study.optimize(objective, n_trials=100, n_jobs=-1)
    except KeyboardInterrupt:
        print("Study interrupted by user. Results so far have been saved.")

    # 4. Print the results
    print("\n" + "=" * 50)
    print("            STUDY COMPLETE            ")
    print("=" * 50)
    print("Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")

    pruned_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.PRUNED]
    )
    complete_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
    )

    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")

    print("\n🏆 Best trial:")
    trial = study.best_trial
    print(f"  Value (Validation Accuracy): {trial.value:.4f}")

    print("  Best Hyperparameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
