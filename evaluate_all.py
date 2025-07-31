# evaluate_all.py
import argparse
import os
import numpy as np
import logging
from sklearn.metrics import r2_score

from src.automl.model import AutoML
from src.automl.preprocessing import Preprocessor
from src.automl.utils import load_data, save_metadata

# Configure logging to monitor progress
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_all_folds(task: str, folds: list[int], output_dir: str, seed: int = 42):
    """
    Evaluates the AutoML pipeline across multiple folds and logs the average R².

    Parameters:
    - task: Dataset name (subfolder in /data)
    - folds: List of fold indices (1-based)
    - output_dir: Where to save predictions and metadata
    - seed: Random seed
    """
    r2_scores = []

    for fold in folds:
        logger.info(f"Evaluating fold {fold} for task '{task}'")

        # Load fold-specific train and test splits
        X_train, y_train, X_test, y_test = load_data(task, fold)

        # === PCA ACTIVATION ===
        preprocessor = Preprocessor(use_pca=True, pca_variance=0.95)

        # Create AutoML pipeline
        automl = AutoML(preprocessing=preprocessor, seed=seed)

        # Train and predict
        automl.fit(X_train, y_train)
        y_pred = automl.predict(X_test)

        # Score and report
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)
        logger.info(f"Fold {fold}: R² score = {r2:.4f}")

        # Save predictions
        os.makedirs(output_dir, exist_ok=True)
        pred_path = os.path.join(output_dir, f"{task}_fold{fold}_pred.npy")
        np.save(pred_path, y_pred)

        # Save metadata (optional)
        save_metadata(automl, task, fold)

    # Log average R² across all folds
    avg_r2 = np.mean(r2_scores)
    logger.info(f"\n✅ Average R² over {len(folds)} folds: {avg_r2:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AutoML across multiple folds.")
    parser.add_argument("--tasks", nargs="+", required=True, help="Dataset names (e.g. bike_sharing_demand)")
    parser.add_argument("--folds", nargs="+", type=int, help="Fold indices (e.g. 1 2 3). Default: All 1–5", default=[1, 2, 3, 4, 5])
    parser.add_argument("--output-dir", type=str, default="out", help="Directory to save predictions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    for task in args.tasks:
        evaluate_all_folds(task, args.folds, args.output_dir, seed=args.seed)
