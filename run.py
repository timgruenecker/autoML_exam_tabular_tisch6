import argparse
import os
import logging
import numpy as np
import pandas as pd
from src.automl.model import AutoML
from src.automl.preprocessing import Preprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_fold_data(task_dir, fold):
    fold_dir = os.path.join(task_dir, str(fold))
    X_train = pd.read_parquet(os.path.join(fold_dir, "X_train.parquet"))
    y_train = pd.read_parquet(os.path.join(fold_dir, "y_train.parquet")).squeeze()
    X_test = pd.read_parquet(os.path.join(fold_dir, "X_test.parquet"))
    return X_train, y_train, X_test


def detect_folds(task_dir):
    """Returns list of fold indices (as int) if fold subdirectories exist"""
    try:
        fold_dirs = [
            int(name) for name in os.listdir(task_dir)
            if os.path.isdir(os.path.join(task_dir, name)) and name.isdigit()
        ]
        return sorted(fold_dirs)
    except FileNotFoundError:
        return []


def main(task: str, fold: int, output_path: str):
    task_path = os.path.join("data", task)
    available_folds = detect_folds(task_path)

    if not available_folds:
        raise ValueError(f"No fold subdirectories found in {task_path}.")

    if fold not in available_folds:
        raise ValueError(f"Requested fold {fold}, but available folds are: {available_folds}")

    logger.info(f"Using fold {fold} for task {task}")
    X_train, y_train, X_test = load_fold_data(task_path, fold)

    preprocessor = Preprocessor()
    automl = AutoML(preprocessing=preprocessor, seed=42)

    logger.info("Fitting AutoML...")
    automl.fit(X_train, y_train)

    logger.info("Generating predictions...")
    y_pred = automl.predict(X_test)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Saving predictions to {output_path}")
    np.save(output_path, y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Name of the dataset (subfolder in 'data/')")
    parser.add_argument("--fold", type=int, required=True, help="Fold number (must exist as subfolder)")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save predictions (.npy)")

    args = parser.parse_args()
    main(args.task, args.fold, args.output_path)
