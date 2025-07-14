from __future__ import annotations
from pathlib import Path
from sklearn.metrics import r2_score
import numpy as np
from automl.data import Dataset
from automl.model import AutoML
import argparse
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Define default data directory relative to this script location
FILE = Path(__file__).absolute().resolve()
DATADIR = FILE.parent / "data"

def main(
    task: str,
    fold: int,
    output_path: Path,
    seed: int,
    datadir: Path,
):
    # Load the dataset for the specified task and fold from the data directory
    dataset = Dataset.load(datadir=datadir, task=task, fold=fold)

    logger.info("Fitting AutoML")

    # Initialize and train the AutoML model with provided seed
    automl = AutoML(random_state=seed)
    automl.fit(dataset.X_train, dataset.y_train)

    # Generate predictions on the test set
    test_preds = automl.predict(dataset.X_test)

    logger.info("Writing predictions to disk")
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save predictions as numpy binary file
    with output_path.open("wb") as f:
        np.save(f, test_preds)

    # Evaluate R² on test labels if available
    if dataset.y_test is not None:
        r2_test = r2_score(dataset.y_test, test_preds)
        logger.info(f"R^2 on test set: {r2_test}")
    else:
        r2_test = None
        logger.info(f"No test labels available for task '{task}'")

    # Save metadata including task info, seed, scores, and timestamp as JSON
    metadata = {
        "task": task,
        "fold": fold,
        "seed": seed,
        "cv_score": getattr(automl, "best_score_", None),
        "test_score": r2_test,
        "best_params": getattr(automl, "best_params_", None),
        "timestamp": datetime.now().isoformat()
    }

    metadata_path = output_path.parent / "metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Metadata written to {metadata_path}")

if __name__ == "__main__":
    # Parse command line arguments for task, fold, output path, seed, and data directory
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True,
                        choices=["bike_sharing_demand", "brazilian_houses", "superconductivity", "wine_quality", "yprop_4_1"])
    parser.add_argument("--output-path", type=Path, default=Path("data/bike_sharing_demand/1/predictions.npy"))
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datadir", type=Path, default=DATADIR)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    # Set logging level based on --quiet flag
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO)

    logger.info(f"Running task {args.task} with fold {args.fold}")
    # Run main function with parsed arguments
    main(args.task, args.fold, args.output_path, args.seed, args.datadir)
