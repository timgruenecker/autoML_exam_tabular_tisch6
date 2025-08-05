import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

import numpy as np
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

import sys
import logging
import time
import json

# === CONSTANTS AND LOGGING SETUP ===
DATA_DIR = Path("../data/exam_dataset/1")
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)  # Ensure logs directory exists
LOG_FILE = LOGS_DIR / "xgboost_hpo_final_pipeline.log"

# Configure logging: both file and stdout for tracking progress and debugging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger()
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(logging.StreamHandler(sys.stdout))

# === LOAD AND PREPROCESS DATA ===
print("[XGBoost HPO] Loading training data...")
X = pd.read_parquet(DATA_DIR / "X_train.parquet")
y = pd.read_parquet(DATA_DIR / "y_train.parquet")["SALE_PRC"]

# XGBoost does not support categorical variables directly: encode as integer codes
categorical_cols = list(X.select_dtypes(include=["category"]).columns)
for col in categorical_cols:
    X[col] = X[col].cat.codes

print(f"[XGBoost HPO] Data loaded. n_samples={len(X)}, n_features={X.shape[1]}, n_cats={len(categorical_cols)}")

# KFold cross-validation setup for reproducibility
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# === DEFINE SEARCH SPACE FOR OPTUNA ===
def get_param_space(bounds=None):
    """Defines hyperparameter search space for XGBoost. Allows overriding default bounds."""
    if bounds is None:
        bounds = {}
    space = {
        "n_estimators": ("int", bounds.get("n_estimators", (1500, 3500))),
        "learning_rate": ("float", bounds.get("learning_rate", (0.01, 0.07))),
        "max_depth": ("int", bounds.get("max_depth", (3, 8))),
        "subsample": ("float", bounds.get("subsample", (0.5, 1.0))),
        "colsample_bytree": ("float", bounds.get("colsample_bytree", (0.5, 1.0))),
        "gamma": ("float", bounds.get("gamma", (0.0, 5.0))),
        "reg_alpha": ("float", bounds.get("reg_alpha", (1e-7, 1e-2))),
        "reg_lambda": ("float", bounds.get("reg_lambda", (1e-7, 1e-2))),
        "min_child_weight": ("float", bounds.get("min_child_weight", (1.0, 10.0))),
    }
    return space

def make_objective(param_space):
    """
    Returns an Optuna objective function that trains XGBoost with CV 
    and returns the mean R^2 score across all folds.
    """
    def objective(trial):
        params = {}
        # Suggest hyperparameters for this trial
        for key, (typ, val) in param_space.items():
            if typ == "int":
                params[key] = trial.suggest_int(key, val[0], val[1])
            elif typ == "float":
                if key in ["learning_rate", "reg_alpha", "reg_lambda"]:
                    params[key] = trial.suggest_float(key, val[0], val[1], log=True)
                else:
                    params[key] = trial.suggest_float(key, val[0], val[1])
        params.update({
            "tree_method": "hist",
            "random_state": 42,
            "early_stopping_rounds": 75,
            "verbosity": 0,
            "n_jobs": 2,  # Use 2 cores per trial for speed
        })

        scores = []
        # Standard KFold cross-validation: train and evaluate model for each fold
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = XGBRegressor(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            preds = model.predict(X_val)
            score = r2_score(y_val, preds)
            scores.append(score)

            # Report intermediate result to Optuna for pruning
            trial.report(score, step=fold_idx)
            # Early pruning if not promising after 2 folds
            if fold_idx < 2 and trial.should_prune():
                print(f"Trial {trial.number} pruned in fold {fold_idx+1}")
                raise optuna.TrialPruned()

        mean_score = float(np.mean(scores))
        print(f"Trial {trial.number}: R2 = {mean_score:.5f}")
        logger.info(f"Trial {trial.number}: R2 = {mean_score:.5f} with params = {params}")
        return mean_score
    return objective

def shrink_param_space(study, param_space, top_frac=0.2):
    """
    Restrict the parameter space based on the top-fraction of best trials so far.
    Useful for narrowing down the search in staged HPO.
    """
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df = df[df["state"] == "COMPLETE"]
    best_df = df.nlargest(int(len(df)*top_frac), "value") if len(df) > 5 else df
    new_space = param_space.copy()
    for key, (typ, val) in param_space.items():
        if typ in ["int", "float"]:
            col = "params_" + key
            best_vals = best_df[col].astype(float)
            lo, hi = best_vals.min(), best_vals.max()
            if lo == hi: hi = lo + (1 if typ == "int" else 1e-4)
            new_space[key] = (typ, (type(val[0])(lo), type(val[1])(hi)))
    return new_space

# === OPTUNA STAGED HPO LOOP ===
def staged_hpo(
    phases=[("Phase 1", 70*60, 0.40), ("Phase 2", 40*60, 0.15), ("Phase 3", 10*60, None)],
    timeout=2*60*60
):
    """
    Run multi-phase Optuna HPO: after each phase, restrict search to the best fraction of parameter space.
    Each phase can have its own time budget and shrink percentage.
    """
    all_study = None
    param_space = get_param_space(bounds={})
    t0 = time.time()
    for pname, sec, frac in phases:
        print(f"\n[XGBoost HPO] === {pname}: {sec//60} min ===")
        logger.info(f"===== {pname}: {sec//60} min =====")
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=HyperbandPruner()
        )
        # Remaining time for this phase
        time_left = timeout - (time.time() - t0)
        phase_time = min(sec, time_left)
        if phase_time <= 0:
            print("[XGBoost HPO] Time budget exceeded, stopping.")
            break
        print(f"[XGBoost HPO] Optimizing for up to {phase_time/60:.1f} min...")
        study.optimize(make_objective(param_space), timeout=phase_time, n_jobs=-1)
        if all_study is None:
            all_study = study
        else:
            all_study.add_trials(study.get_trials(deepcopy=True))
        # After this phase: restrict parameter search if requested
        if frac is not None and len(study.trials) > 2:
            param_space = shrink_param_space(study, param_space, top_frac=frac)
            print(f"[XGBoost HPO] Parameter space shrunk: {param_space}")
            logger.info(f"Parameter space shrunk to: {param_space}")

    # === Summary and output of best found parameters ===
    print("\n[XGBoost HPO] === Best results from all phases ===")
    logger.info("==== Best results from all phases ====")
    print(f"[XGBoost HPO] Best R2: {all_study.best_value:.5f}")
    logger.info(f"Best R2: {all_study.best_value:.5f}")
    print("[XGBoost HPO] Best Params:")
    for k, v in all_study.best_params.items():
        print(f"    {k}: {v}")
        logger.info(f"  {k}: {v}")

    out = {
        "best_score": all_study.best_value,
        "best_params": all_study.best_params,
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/xgboost_best_params.json", "w") as f:
        json.dump(out, f, indent=2)
    print("[XGBoost HPO] Done! Best parameters saved to results/xgboost_best_params.json.")

if __name__ == "__main__":
    # === Start staged Optuna HPO ===
    print("\n[XGBoost HPO] Starting XGBoost hyperparameter optimization...")
    staged_hpo([
        ("Phase 1", 70*60, 0.40),   # 70 min, shrink to top 40%
        ("Phase 2", 40*60, 0.15),   # 40 min, shrink to top 15%
        ("Phase 3", 10*60, None)    # 10 min, no further shrink
    ], timeout=2*60*60)
    print("[XGBoost HPO] XGBoost HPO complete.")

    # === FINAL OOF PREDICTIONS WITH BEST PARAMS ===
    print("\n[XGBoost HPO] Generating final OOF predictions with best parameters...")
    DATA_DIR = Path("../data/exam_dataset/1")
    OOF_DIR = Path("results/oof")
    X = pd.read_parquet(DATA_DIR / "X_train.parquet")
    y = pd.read_parquet(DATA_DIR / "y_train.parquet")["SALE_PRC"]

    # Encode categoricals again (consistency with HPO loop)
    cat_cols = list(X.select_dtypes(include=["category"]).columns)
    if cat_cols:
        for col in cat_cols:
            X[col] = X[col].cat.codes

    oof_pred = np.zeros(len(y))
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Load best parameters from Optuna
    with open("results/xgboost_best_params.json", "r") as f:
        best_dict = json.load(f)
    best_params = best_dict["best_params"]

    # Train OOF models: standard CV loop, save OOF predictions
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        print(f"[XGBoost HPO] Fold {fold}")
        model = XGBRegressor(**best_params, random_state=42)
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
            verbose=False
        )
        pred = model.predict(X.iloc[va_idx])
        oof_pred[va_idx] = pred
        score = r2_score(y.iloc[va_idx], pred)
        print(f"  R² = {score:.5f}")

    # Store OOF predictions to disk for later ensembling
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    df_oof = pd.DataFrame({"oof_xgb_hpo": oof_pred}, index=X.index)
    df_oof.to_parquet(OOF_DIR / "oof_xgb_hpo.parquet")
    print(f"[XGBoost HPO] OOF predictions saved to: {OOF_DIR / 'oof_xgb_hpo.parquet'}")