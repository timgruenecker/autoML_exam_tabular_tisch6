# run.py
import argparse
import logging
import numpy as np
import os

from src.automl.model import AutoML
from src.automl.preprocessing import Preprocessor
from src.automl.utils import load_data, save_metadata

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(task: str, fold: int, output_path: str, seed: int = 42):
    """
    Entry point for running the AutoML system on a single fold.

    Args:
        task (str): Dataset name
        fold (int): Fold index (starting from 1)
        output_path (str): Path to save prediction output (NumPy file)
        seed (int): Random seed
    """
    logger.info(f"Running task '{task}', fold {fold}...")

    # Load training and test data
    X_train, y_train, X_test, y_test = load_data(task, fold)

    # Initialize preprocessing with PCA enabled
    preprocessor = Preprocessor(use_pca=True, pca_variance=0.95)

    # Initialize AutoML instance
    automl = AutoML(preprocessing=preprocessor, seed=seed)

    # Fit model on training data
    automl.fit(X_train, y_train)

    # Generate predictions
    logger.info("Generating predictions...")
    y_pred = automl.predict(X_test)

    # Save predictions
    logger.info(f"Saving predictions to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, y_pred)

    # Save metadata
    save_metadata(automl, task, fold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AutoML on a single fold.")
    parser.add_argument("--task", type=str, required=True, help="Name of dataset")
    parser.add_argument("--fold", type=int, required=True, help="Fold number (starting from 1)")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save predictions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()
    main(args.task, args.fold, args.output_path, seed=args.seed)
