import argparse
import os
import numpy as np
import logging
from sklearn.metrics import r2_score
import glob

from src.automl.model import AutoML
from src.automl.preprocessing import Preprocessor
from src.automl.utils import load_data, save_metadata

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_all_folds(task: str, folds: list[int], output_dir: str, seed: int = 42):
    """
    Evaluates the AutoML pipeline for a given task across all specified folds.

    Args:
        task (str): Name of the dataset (folder name in /data)
        folds (list[int]): List of fold indices (starting from 1)
        output_dir (str): Directory to store predictions and metadata
        seed (int): Random seed for reproducibility
    """
    r2_scores = []

    for fold in folds:
        logger.info(f"Evaluating fold {fold} for task '{task}'")

        # Load training and test sets for this fold
        X_train, y_train, X_test, y_test = load_data(task, fold)

        # Initialize preprocessing pipeline with PCA
        preprocessor = Preprocessor(use_pca=True, pca_variance=0.95)

        # Create AutoML instance with specified seed
        automl = AutoML(preprocessing=preprocessor, seed=seed)

        # Fit model and predict
        automl.fit(X_train, y_train)
        y_pred = automl.predict(X_test)

        # Compute R² score for this fold
        if y_test is not None:
            r2 = r2_score(y_test, y_pred)
            r2_scores.append(r2)
            logger.info(f"Fold {fold}: R² score = {r2:.4f}")
        else:
            logger.warning(f"No ground truth available for fold {fold}. Skipping R² computation.")

        # Save predictions to file
        os.makedirs(output_dir, exist_ok=True)
        pred_path = os.path.join(output_dir, f"{task}_fold{fold}_pred.npy")
        np.save(pred_path, y_pred)

        # Save associated metadata
        save_metadata(automl, task, fold)

    # Report average performance across all folds
    avg_r2 = np.mean(r2_scores)
    logger.info(f"\n✅ Average R² over {len(folds)} folds: {avg_r2:.4f}")

def infer_folds(task: str):
    """
    Automatically determines the available folds for a dataset by scanning its directory.

    Args:
        task (str): Name of dataset (i.e., name of folder in /data)

    Returns:
        List[int]: List of detected fold indices
    """
    path = os.path.join("data", task)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such task directory: {path}")

    # Extract all fold directories that are numeric
    folds = [
        int(os.path.basename(f))
        for f in glob.glob(os.path.join(path, "*"))
        if os.path.isdir(f) and os.path.basename(f).isdigit()
    ]

    if not folds:
        raise ValueError(f"No valid folds found for task '{task}' in {path}")

    return sorted(folds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AutoML across multiple folds.")
    parser.add_argument("--tasks", nargs="+", required=True, help="Dataset names (e.g. bike_sharing_demand)")
    parser.add_argument("--folds", nargs="+", type=int, help="Fold indices (e.g. 1 2 3). If omitted, all folds are inferred automatically.")
    parser.add_argument("--output-dir", type=str, default="out", help="Directory to store predictions and metadata")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    for task in args.tasks:
        # Automatically infer folds if not specified
        folds = args.folds if args.folds else infer_folds(task)
        logger.info(f"Detected folds for task '{task}': {folds}")
        evaluate_all_folds(task, folds, args.output_dir, seed=args.seed)
