import logging
import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

from .registry import ModelRegistry
from .preprocessing import load_data, preprocess_data

logger = logging.getLogger(__name__)

def plot_model_scores(results_df, output_dir, timestamp):
    """Creates a bar plot comparing mean R² scores of all models."""
    plt.figure(figsize=(10, 6))
    plt.bar(results_df["model"], results_df["mean_r2"], yerr=results_df["std_r2"], capsize=5)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean R² (CV)")
    plt.title("Model Comparison")
    plt.tight_layout()

    plot_path = Path(output_dir) / f"model_comparison_{timestamp}.png"
    plt.savefig(plot_path)
    logger.info(f"Saved model comparison plot to {plot_path}")
    plt.close()

def evaluate_all(train_path, test_path=None, target_column="target", output_dir="outputs", cv=5):
    """Evaluates all registered models and saves the best one."""
    logger.info("Starting model evaluation...")

    df_train = load_data(train_path)
    X_train, y_train = preprocess_data(df_train, target_column=target_column)

    if test_path:
        df_test = load_data(test_path)
        X_test, y_test = preprocess_data(df_test, target_column=target_column)
    else:
        X_test, y_test = None, None

    registry = ModelRegistry()
    models = registry.get_models()
    results = []

    for name, model in models:
        logger.info(f"Evaluating model: {name}")
        try:
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
            mean_r2 = scores.mean()
            std_r2 = scores.std()
            logger.info(f"{name} R²: {mean_r2:.4f} ± {std_r2:.4f}")
            results.append({
                "model": name,
                "mean_r2": mean_r2,
                "std_r2": std_r2,
                "instance": model
            })
        except Exception as e:
            logger.error(f"Error with model {name}: {e}")

    results_df = pd.DataFrame(results).sort_values("mean_r2", ascending=False)
    if results_df.empty:
        logger.warning("No models were successfully evaluated.")
        return

    best_row = results_df.iloc[0]
    best_model = best_row["instance"]
    best_name = best_row["model"]
    logger.info(f"Best model: {best_name} with R² = {best_row['mean_r2']:.4f}")

    best_model.fit(X_train, y_train)

    if X_test is not None and y_test is not None:
        y_pred = best_model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        logger.info(f"Test R² for {best_name}: {test_r2:.4f}")
    else:
        test_r2 = None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path(output_dir).mkdir(exist_ok=True)
    model_path = Path(output_dir) / f"{best_name}_{timestamp}.pkl"
    joblib.dump(best_model, model_path)
    logger.info(f"Saved best model to: {model_path}")

    results_df.to_csv(Path(output_dir) / f"evaluation_{timestamp}.csv", index=False)
    logger.info("Saved evaluation results to CSV.")

    plot_model_scores(results_df, output_dir, timestamp)
    logger.info("Model evaluation complete.")
