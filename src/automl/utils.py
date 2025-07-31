import pandas as pd
import os
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_data(task_name: str, fold: int):
    """
    Loads X_train, X_test, y_train, y_test from the specified dataset folder.

    Args:
        task_name (str): Name of the dataset (e.g. 'wine_quality')
        fold (int): Fold number (starting at 1)

    Returns:
        Tuple of pandas DataFrames: X_train, y_train, X_test, y_test
    """
    base_path = os.path.join("data", task_name, str(fold))
    X_train = pd.read_parquet(os.path.join(base_path, "X_train.parquet"))
    y_train = pd.read_parquet(os.path.join(base_path, "y_train.parquet")).squeeze()
    X_test = pd.read_parquet(os.path.join(base_path, "X_test.parquet"))
    y_test_path = os.path.join(base_path, "y_test.parquet")
    y_test = pd.read_parquet(y_test_path).squeeze() if os.path.exists(y_test_path) else None
    return X_train, y_train, X_test, y_test


def save_metadata(automl, task: str, fold: int):
    """
    Extracts and saves relevant metadata from the AutoML run to a JSON file.

    Args:
        automl (AutoML): Trained AutoML instance
        task (str): Task name
        fold (int): Fold number
    """
    def make_serializable(d):
        # Konvertiert alle numpy-Typen zu Python-Typen rekursiv
        if isinstance(d, dict):
            return {k: make_serializable(v) for k, v in d.items()}
        elif isinstance(d, (np.integer, np.int64)):
            return int(d)
        elif isinstance(d, (np.floating, np.float64)):
            return float(d)
        elif isinstance(d, (np.ndarray, list, tuple)):
            return [make_serializable(i) for i in d]
        else:
            return d

    meta_features = getattr(automl.preprocessing, "meta_features_", {})
    metadata = {
        "task": task,
        "fold": int(fold),
        "models_used": list(automl.models.keys()) if hasattr(automl, "models") else [],
        "meta_features": make_serializable(meta_features),
    }

    path = os.path.join("out", f"{task}_fold{fold}_metadata.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Saved metadata to {path}")

