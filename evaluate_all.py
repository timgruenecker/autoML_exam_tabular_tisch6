import os
import numpy as np
import argparse
import pandas as pd
from sklearn.metrics import r2_score
from src.automl.model import AutoML
from src.automl.preprocessing import Preprocessor


def load_data(task, fold):
    # Load train/test data for given task and fold
    base_path = os.path.join("data", task, str(fold))
    X_train = pd.read_parquet(os.path.join(base_path, "X_train.parquet"))
    y_train = pd.read_parquet(os.path.join(base_path, "y_train.parquet")).values.ravel()
    X_test = pd.read_parquet(os.path.join(base_path, "X_test.parquet"))
    y_test_path = os.path.join(base_path, "y_test.parquet")
    y_test = None
    if os.path.exists(y_test_path):
        y_test = pd.read_parquet(y_test_path).values.ravel()
    return X_train, y_train, X_test, y_test


def main(tasks, folds, output_dir, seed=42):
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for task in tasks:
        for fold in folds:
            print(f"\n=== Evaluating task: {task}, fold: {fold} ===")
            X_train, y_train, X_test, y_test = load_data(task, fold)

            preprocessor = Preprocessor(
                use_pca=args.use_pca,
                n_components=args.pca_components
            )

            automl = AutoML(preprocessing=preprocessor, seed=args.seed)
            automl.fit(X_train, y_train)
            y_pred = automl.predict(X_test)

            # Save predictions
            pred_path = os.path.join(output_dir, f"preds_{task}_fold{fold}.npy")
            np.save(pred_path, y_pred)
            print(f"Saved predictions to {pred_path}")

            # Evaluate if y_test is available
            if y_test is not None:
                score = r2_score(y_test, y_pred)
                print(f"R2 score on test set: {score:.4f}")
                results.append({"task": task, "fold": fold, "r2_score": score})
            else:
                print("No ground truth available for test set, skipping evaluation.")

    # Save summary results if any
    if results:
        import json
        summary_path = os.path.join(output_dir, "evaluation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nSummary of evaluations saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AutoML pipeline on multiple datasets and folds")
    parser.add_argument("--tasks", nargs="+", required=True,
                        help="List of tasks to evaluate (e.g. bike_sharing_demand wine_quality)")
    parser.add_argument("--folds", nargs="+", type=int, default=[1], help="List of folds to evaluate")
    parser.add_argument("--output-dir", default="out", help="Directory to save predictions and evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument('--use-pca', action='store_true', help="Enable PCA in preprocessing")
    parser.add_argument('--pca-components', type=float, default=0.95, help="Number of PCA components (variance ratio)")
    args = parser.parse_args()

    main(args.tasks, args.folds, args.output_dir, args.seed)
