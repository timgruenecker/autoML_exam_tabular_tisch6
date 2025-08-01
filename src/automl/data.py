from __future__ import annotations
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    """
    Structured container for train/test data of one fold.
    """
    path: Path
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series | None = None

    @classmethod
    def load(cls, datadir: Path, task: str, fold: int) -> Dataset:
        """
        Loads the dataset for a given task and fold.

        Args:
            datadir (Path): Root data directory (e.g. 'data/')
            task (str): Name of dataset folder
            fold (int): Fold number (1-based)

        Returns:
            Dataset: A structured dataclass with training and test data
        """
        # Special handling for exam dataset (only 1 fold)
        if task == "exam_dataset":
            path = datadir / task / str(1)
        else:
            path = datadir / task / str(fold)

        if not path.exists():
            raise FileNotFoundError(path)

        # Define expected file paths
        X_train_path = path / "X_train.parquet"
        y_train_path = path / "y_train.parquet"
        X_test_path = path / "X_test.parquet"
        y_test_path = path / "y_test.parquet"

        # Load parquet files and return as a dataclass
        return Dataset(
            path=path,
            X_train=pd.read_parquet(X_train_path),
            y_train=pd.read_parquet(y_train_path).iloc[:, 0],
            X_test=pd.read_parquet(X_test_path),
            y_test=pd.read_parquet(y_test_path).iloc[:, 0] if y_test_path.exists() else None,
        )
