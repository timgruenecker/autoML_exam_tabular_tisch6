# run.py
import argparse
import logging
import numpy as np
import pandas as pd
import os

from src.automl.model import AutoML
from src.automl.preprocessing import Preprocessor
from src.automl.utils import load_data, save_metadata

# Set up logging to display progress and debugging info
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(task: str, fold: int, output_path: str, seed: int = 42):
    """
    Main function to train AutoML model and generate predictions for one specific fold.

    Parameters:
    - task: Name of the dataset (subfolder in /data)
    - fold: Which outer fold (1-based index)
    - output_path: Path to store the prediction file (e.g., 'out/preds.npy')
    - seed: Random seed for reproducibility
    """
    logger.info(f"Running task '{task}', fold {fold}...")

    # Load training and test data for the selected fold
    X_train, y_train, X_test, y_test = load_data(task, fold)

    # === PCA ACTIVATION ===
    # Initialize preprocessing pipeline with PCA enabled (adjust variance if needed)
    preprocessor = Preprocessor(use_pca=True, pca_variance=0.95)

    # Initialize AutoML system with preprocessing and random seed
    automl = AutoML(preprocessing=preprocessor, seed=seed)

    # Fit AutoML pipeline on training data
    automl.fit(X_train, y_train)

    # Generate predictions for X_test
    logger.info("Generating predictions...")
    y_pred = automl.predict(X_test)

    # Save predictions to the specified output path
    logger.info(f"Saving predictions to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, y_pred)

    # Save metadata (e.g. model info, meta-features)
    save_metadata(automl.get_metadata(), task, fold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AutoML on a single fold.")
    parser.add_argument("--task", type=str, required=True, help="Name of dataset")
    parser.add_argument("--fold", type=int, required=True, help="Fold number (starting from 1)")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save predictions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()
    main(args.task, args.fold, args.output_path, seed=args.seed)
