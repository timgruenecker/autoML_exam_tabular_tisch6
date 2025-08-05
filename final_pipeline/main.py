import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
import json
import subprocess
import shutil

# === CONSTANTS & PATHS ===
DATA_DIR = Path("../data/exam_dataset/1")
X_TRAIN_FILE = "X_train.parquet"
Y_TRAIN_FILE = "y_train.parquet"
RESULTS_DIR = Path("results")
OOF_DIR = Path("results/oof")
BASELINE_SCORES_FILE = RESULTS_DIR / "baseline_scores.json"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def save_json(data, path: Path):
    """Save dictionary to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_data(data_path: Path):
    """Load X and y data from Parquet files."""
    X = pd.read_parquet(data_path / X_TRAIN_FILE)
    y = pd.read_parquet(data_path / Y_TRAIN_FILE)["SALE_PRC"]
    return X, y

def detect_column_types(X: pd.DataFrame):
    """Detect categorical and numerical columns."""
    categorical = X.select_dtypes(include="category").columns.tolist()
    numerical = X.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()
    return categorical, numerical

def get_models():
    """Return dict of all candidate baseline models."""
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(),
        "XGBoost": XGBRegressor(verbosity=0),
        "LightGBM": LGBMRegressor(verbosity=-1),
        "CatBoost": CatBoostRegressor(verbose=0, train_dir="catboost_tmp"),
    }

def encode_for_gbdt(X: pd.DataFrame, categorical_cols: list):
    """
    Encode categorical columns for models (GBDT, RF, LR, etc.)
    CatBoost and LightGBM can natively handle categories; XGBoost, RF, LR need integer encoding.
    """
    X_enc = X.copy()
    for col in categorical_cols:
        X_enc[col] = X_enc[col].astype("category").cat.codes
    return X_enc

def cast_categories(X: pd.DataFrame, categorical_cols: list):
    """
    Cast columns to 'category' dtype (used for LightGBM).
    """
    X_casted = X.copy()
    for col in categorical_cols:
        X_casted[col] = X_casted[col].astype("category")
    return X_casted

def run_oof_predictions(X, y, categorical_cols, models, cv):
    """
    Compute and save OOF (out-of-fold) predictions for all base models.
    Results (R² per fold, mean/std, and full OOF vector) are saved for each model.
    """
    oof_scores = {}
    OOF_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        print(f"\n== {name} OOF ==")
        oof_pred = np.zeros(len(y))
        fold_scores = []

        for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
            print(f"  Fold {fold+1}", end=" ")

            # Different model types require different handling of categorical columns
            if name == "CatBoost":
                X_train, X_valid = X.iloc[tr_idx], X.iloc[va_idx]
                cat_indices = [X.columns.get_loc(c) for c in categorical_cols]
                model_fold = CatBoostRegressor(verbose=0, train_dir="catboost_tmp")
                model_fold.fit(X_train, y.iloc[tr_idx], cat_features=cat_indices)
                preds = model_fold.predict(X_valid)
            elif name == "LightGBM":
                X_train = cast_categories(X.iloc[tr_idx], categorical_cols)
                X_valid = cast_categories(X.iloc[va_idx], categorical_cols)
                model_fold = LGBMRegressor(verbosity=-1)
                model_fold.fit(X_train, y.iloc[tr_idx])
                preds = model_fold.predict(X_valid)
            elif name in ["XGBoost", "LinearRegression", "RandomForest"]:
                X_train = encode_for_gbdt(X.iloc[tr_idx], categorical_cols)
                X_valid = encode_for_gbdt(X.iloc[va_idx], categorical_cols)
                if name == "XGBoost":
                    model_fold = XGBRegressor(verbosity=0)
                elif name == "RandomForest":
                    model_fold = RandomForestRegressor()
                else:
                    model_fold = LinearRegression()
                model_fold.fit(X_train, y.iloc[tr_idx])
                preds = model_fold.predict(X_valid)
            else:
                # Fallback for any unexpected case
                X_train, X_valid = X.iloc[tr_idx], X.iloc[va_idx]
                model_fold = model.__class__()  # instantiate new
                model_fold.fit(X_train, y.iloc[tr_idx])
                preds = model_fold.predict(X_valid)

            oof_pred[va_idx] = preds
            fold_score = r2_score(y.iloc[va_idx], preds)
            fold_scores.append(fold_score)
            print(f"R² = {fold_score:.4f}")

        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        print(f"OOF {name}: R² mean = {mean_score:.4f} ± {std_score:.4f}")

        oof_scores[name] = {"mean": mean_score, "std": std_score}

        df_oof = pd.DataFrame({f"oof_{name.lower()}": oof_pred}, index=X.index)
        df_oof.to_parquet(OOF_DIR / f"oof_{name.lower()}.parquet")

    save_json(oof_scores, BASELINE_SCORES_FILE)
    print(f"\nAll OOF Parquet files written to: {OOF_DIR}")
    return oof_scores

def compare_and_select_best_oof(model_name: str, y_true):
    """
    Compares baseline OOF and HPO OOF for a given model,
    selects the best one as oof_<model>_final.parquet and removes unused files.
    Returns R² of both for reporting.
    """
    base_path = OOF_DIR / f"oof_{model_name.lower()}.parquet"
    hpo_path = OOF_DIR / f"oof_{model_name.lower()}_hpo.parquet"
    final_path = OOF_DIR / f"oof_{model_name.lower()}_final.parquet"

    if not hpo_path.exists():
        print(f"[WARN] No HPO OOF file for {model_name}. Skipping comparison.")
        shutil.copy(base_path, final_path)
        try:
            base_path.unlink()
        except Exception:
            pass
        return None

    oof_base = pd.read_parquet(base_path).iloc[:,0].values
    oof_hpo = pd.read_parquet(hpo_path).iloc[:,0].values

    score_base = r2_score(y_true, oof_base)
    score_hpo = r2_score(y_true, oof_hpo)

    if score_hpo > score_base:
        print(f"⚡️ HPO for {model_name} is better ({score_hpo:.5f} > {score_base:.5f}), using HPO result.")
        shutil.copy(hpo_path, final_path)
    else:
        print(f"Baseline for {model_name} is better ({score_base:.5f} >= {score_hpo:.5f}), using baseline result.")
        shutil.copy(base_path, final_path)

    # Remove old files after comparison
    for path in [base_path, hpo_path]:
        try:
            path.unlink()
        except Exception:
            pass

    return {"model": model_name, "baseline_r2": score_base, "hpo_r2": score_hpo}

def main():
    # 1. Load data and detect column types
    X, y = load_data(DATA_DIR)
    categorical_cols, _ = detect_column_types(X)
    models = get_models()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # 2. Compute OOF predictions for all models
    oof_scores = run_oof_predictions(X, y, categorical_cols, models, cv)

    # 3. Select the best baseline model (highest mean OOF R²)
    best_model = max(oof_scores.items(), key=lambda x: x[1]["mean"])[0]
    print(f"\nBest baseline model (highest mean OOF R²): {best_model}")

    # 4. Run HPO for the best model (if script is available)
    hpo_scripts = {
        "CatBoost": "catboost_hpo.py",
        "XGBoost": "xgb_hpo.py",
        "LightGBM": "lgbm_hpo.py",
        "RandomForest": "rf_hpo.py"
    }

    hpo_script = hpo_scripts.get(best_model)
    if hpo_script is not None:
        print(f"\nLaunching HPO for {best_model} via {hpo_script}...")
        subprocess.run(["python", hpo_script], check=True)
    else:
        print(f"No HPO script found for '{best_model}'. Skipping HPO step.")

    # 5. After HPO: Compare baseline and HPO OOF and save the best as final OOF
    print(f"\nComparing OOF predictions for {best_model} (baseline vs. HPO)...")
    compare_and_select_best_oof(best_model, y)

    # 6. Launch knowledge distillation with TabNet
    print(f"\nLaunching Knowledge Distillation (kd_tabnet.py)...")
    subprocess.run(["python", "kd_tabnet.py"], check=True)

    # 7. Launch ensembling step
    print(f"\nLaunching ensembling step (ensembling.py)...")
    subprocess.run(["python", "ensembling.py"], check=True)

    print(f"\nThe submission CSV has been generated and the entire pipeline is now complete.")

if __name__ == "__main__":
    main()