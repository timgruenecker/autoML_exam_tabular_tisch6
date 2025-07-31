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


def save_metadata(metadata: dict, task: str, fold: int):
    """
    Saves a dictionary of metadata to a JSON file under ./out/metadata/{task}/fold_{fold}.json

    Args:
        metadata (dict): Metadata dictionary to save
        task (str): Name of the dataset
        fold (int): Fold number
    """
    path = os.path.join("out", "metadata", task, f"fold_{fold}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Saved metadata to {path}")