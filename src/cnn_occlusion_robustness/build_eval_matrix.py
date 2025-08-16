import pandas as pd
import json
import os
import glob
import argparse
from datetime import datetime


def build_matrix(eval_dir: str, output_dir: str):
    """
    Collects individual evaluation JSON files and builds a comprehensive
    performance matrix as a CSV file.
    """
    # Find all evaluation result files
    json_files = glob.glob(os.path.join(eval_dir, "*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"No evaluation .json files found in '{eval_dir}'. "
            "Please run eval.py first."
        )

    print(f"Found {len(json_files)} evaluation result files.")

    # --- 1. Parse Data from Files ---
    results = []
    for f_path in json_files:
        with open(f_path, "r") as f:
            data = json.load(f)

            # Extract the training condition from the model path
            # e.g., 'results/models/light_rain_model.pth' -> 'light_rain'
            model_name = os.path.basename(data["model_path"])
            train_condition = model_name.replace("_model.pth", "")

            results.append(
                {
                    "train_condition": train_condition,
                    "test_condition": data["test_effect"],
                    "accuracy": data["accuracy"],
                }
            )

    # --- 2. Create and Pivot the DataFrame ---
    df = pd.DataFrame(results)

    # Pivot the table to get the desired matrix format
    # Rows: Training Conditions, Columns: Test Conditions, Values: Accuracy
    try:
        eval_matrix = df.pivot(
            index="train_condition", columns="test_condition", values="accuracy"
        )
    except Exception as e:
        print(
            "Error pivoting DataFrame. This can happen if you have duplicate "
            "(train_condition, test_condition) pairs."
        )
        print(f"DataFrame Head:\n{df.head()}")
        print(
            f"Duplicate entries:\n{df[df.duplicated(subset=['train_condition', 'test_condition'], keep=False)]}"
        )
        raise e

    # --- 3. Save the Matrix ---
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"evaluation_matrix_{timestamp}.csv")

    eval_matrix.to_csv(output_path)

    print("\nEvaluation Matrix:")
    print(eval_matrix.to_string(float_format="%.4f"))
    print(f"\n✅ Successfully saved evaluation matrix to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build an evaluation matrix from individual result files."
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        required=True,
        help="Directory containing the JSON results from eval.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory where the final evaluation_matrix.csv will be saved.",
    )
    args = parser.parse_args()

    build_matrix(args.eval_dir, args.output_dir)
