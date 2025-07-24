import argparse
import numpy as np
from pathlib import Path
import logging

from src.automl.data import Dataset
from src.automl.preprocessing import build_preprocessing_pipeline
from src.automl.model import AutoML
import os

logging.basicConfig(level=logging.INFO)

def main(task, fold, output_path):
    logger = logging.getLogger(__name__)
    dataset = Dataset.load(Path("data"), task=task, fold=fold)
    X_train, y_train = dataset.X_train, dataset.y_train
    X_test = dataset.X_test

    preprocessing = build_preprocessing_pipeline(X_train)
    automl = AutoML(preprocessing=preprocessing)

    logger.info("Fitting AutoML model...")
    automl.fit(X_train, y_train)

    logger.info("Generating predictions...")
    y_pred = automl.predict(X_test)

    logger.info(f"Saving predictions to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Name of the task (e.g. bike_sharing_demand)")
    parser.add_argument("--fold", type=int, default=1, help="Fold number to use")
    parser.add_argument("--output-path", type=str, required=True, help="Where to save predictions (as .npy)")
    args = parser.parse_args()

    main(args.task, args.fold, args.output_path)
