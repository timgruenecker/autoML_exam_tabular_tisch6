import logging
from pathlib import Path
import pandas as pd
import sklearn.metrics
from datetime import datetime
import joblib
import matplotlib.pyplot as plt

from src.automl.data import Dataset
from src.automl.model import AutoML

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

def plot_scores(results_df: pd.DataFrame, output_dir: Path, timestamp: str):
    plt.figure(figsize=(8, 5))
    plt.bar(results_df["fold"], results_df["r2_score"])
    plt.xlabel("Fold")
    plt.ylabel("R² Score")
    plt.title("R² Scores per Fold")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_path = output_dir / f"fold_scores_{timestamp}.png"
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Saved fold scores plot to {plot_path}")

def evaluate_all_folds(datadir: Path, task: str, output_dir: Path):
    logging.info(f"Starting evaluation for all folds in task '{task}' under {datadir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    task_dir = datadir / task
    if not task_dir.exists() or not task_dir.is_dir():
        raise FileNotFoundError(f"Task directory does not exist: {task_dir}")

    fold_dirs = sorted([p for p in task_dir.iterdir() if p.is_dir()])
    if not fold_dirs:
        raise RuntimeError(f"No fold subdirectories found in {task_dir}")

    all_results = []

    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        logging.info(f"Processing fold: {fold_name}")

        try:
            fold_number = int(fold_name)
        except ValueError:
            logging.warning(f"Skipping folder {fold_name} because its name is not a number.")
            continue

        dataset = Dataset.load(datadir, task, fold_number)

        model = AutoML()
        model.fit(dataset.X_train, dataset.y_train)

        if dataset.X_test is not None and dataset.y_test is not None:
            y_pred = model.predict(dataset.X_test)
            r2 = sklearn.metrics.r2_score(dataset.y_test, y_pred)
            logging.info(f"Fold {fold_name} R² score: {r2:.4f}")
        else:
            r2 = None
            logging.warning(f"No test labels available for fold {fold_name}. Skipping test evaluation.")

        # Optional: Modell speichern
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = output_dir / f"automl_model_fold{fold_name}_{timestamp}.pkl"
        joblib.dump(model, model_path)
        logging.info(f"Saved model for fold {fold_name} to {model_path}")

        all_results.append({"fold": fold_name, "r2_score": r2})

    # Ergebnisse zusammenfassen und speichern
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(all_results)
    csv_path = output_dir / f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(csv_path, index=False)
    logging.info(f"Saved evaluation summary CSV to {csv_path}")

    plot_scores(results_df, output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    logging.info("Evaluation completed.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate AutoML model on all folds of a dataset task.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Base data directory, e.g. 'data'")
    parser.add_argument("--task", type=str, required=True, help="Task name, e.g. 'bike_sharing_demand'")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"), help="Directory to save models and results")

    args = parser.parse_args()

    evaluate_all_folds(args.data_dir, args.task, args.output_dir)
