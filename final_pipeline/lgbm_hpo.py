import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor, early_stopping
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
import sys
import logging
import time
import json
import numpy as np

# === CONSTANTS & LOGGING SETUP ===
DATA_DIR = Path("../data/exam_dataset/1")
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "lgbm_hpo_final_pipeline.log"

# Configure logging: both file and stdout for monitoring progress and debugging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger()
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(logging.StreamHandler(sys.stdout))

# === LOAD DATA ===
print("[LGBM HPO] Loading training data...")
X = pd.read_parquet(DATA_DIR / "X_train.parquet")
y = pd.read_parquet(DATA_DIR / "y_train.parquet")["SALE_PRC"]
categorical_cols = list(X.select_dtypes(include=["category"]).columns)
print(f"[LGBM HPO] Data loaded. n_samples={len(X)}, n_features={X.shape[1]}, n_cats={len(categorical_cols)}")

# 5-fold cross-validation for robust evaluation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# === DEFINE SEARCH SPACE FOR OPTUNA ===
def get_param_space(bounds=None):
    """Defines the LightGBM hyperparameter search space for Optuna HPO."""
    if bounds is None:
        bounds = {}
    space = {
        "n_estimators": ("int", bounds.get("n_estimators", (1500, 3500))),
        "learning_rate": ("float", bounds.get("learning_rate", (0.01, 0.07))),
        "max_depth": ("int", bounds.get("max_depth", (3, 12))),
        "num_leaves": ("int", bounds.get("num_leaves", (15, 256))),
        "min_child_samples": ("int", bounds.get("min_child_samples", (1, 200))),
        "subsample": ("float", bounds.get("subsample", (0.5, 1.0))),
        "colsample_bytree": ("float", bounds.get("colsample_bytree", (0.5, 1.0))),
        "reg_alpha": ("float", bounds.get("reg_alpha", (1e-7, 1e-1))),
        "reg_lambda": ("float", bounds.get("reg_lambda", (1e-7, 1e-1))),
    }
    return space

def make_objective(param_space):
    """
    Returns an Optuna objective function that runs KFold CV for LGBM with a given parameter set.
    Early-stopping and pruning enabled for efficiency.
    """
    def objective(trial):
        params = {}
        # Suggest hyperparameters for current trial
        for key, (typ, val) in param_space.items():
            if typ == "int":
                params[key] = trial.suggest_int(key, val[0], val[1])
            elif typ == "float":
                if key in ["learning_rate", "reg_alpha", "reg_lambda"]:
                    params[key] = trial.suggest_float(key, val[0], val[1], log=True)
                else:
                    params[key] = trial.suggest_float(key, val[0], val[1])
        params.update({
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": 2,  # Use 2 CPU cores per trial
        })

        scores = []
        # Cross-validation loop
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Categorical columns must be 'category' dtype for LightGBM
            for col in categorical_cols:
                X_train[col] = X_train[col].astype("category")
                X_val[col] = X_val[col].astype("category")

            model = LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[early_stopping(stopping_rounds=75, verbose=False)],
                categorical_feature=categorical_cols if categorical_cols else "auto"
            )
            preds = model.predict(X_val)
            score = r2_score(y_val, preds)
            scores.append(score)

            # Report progress to Optuna for pruning
            trial.report(score, step=fold_idx)
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
    Shrinks the search space based on the top-fraction of best Optuna trials so far.
    This allows for more focused search in the next phase.
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

# === STAGED, TIME-BASED HPO LOOP ===
def staged_hpo(
    phases=[("Phase 1", 55*60, 0.40), ("Phase 2", 43*60, 0.15), ("Phase 3", 7*60, None)],
    timeout=2*60*60
):
    """
    Runs multi-phase Optuna HPO for LightGBM.
    After each phase, shrinks the parameter search space to the top-fraction of best trials.
    Each phase can have its own time and shrink fraction.
    """
    all_study = None
    param_space = get_param_space(bounds={})
    t0 = time.time()
    for pname, sec, frac in phases:
        print(f"\n[LGBM HPO] === {pname}: {sec//60} min ===")
        logger.info(f"===== {pname}: {sec//60} min =====")
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=HyperbandPruner()
        )
        time_left = timeout - (time.time() - t0)
        phase_time = min(sec, time_left)
        if phase_time <= 0:
            print("[LGBM HPO] Time budget exceeded, stopping.")
            break
        print(f"[LGBM HPO] Optimizing for up to {phase_time/60:.1f} min...")
        study.optimize(make_objective(param_space), timeout=phase_time, n_jobs=-1)
        if all_study is None:
            all_study = study
        else:
            all_study.add_trials(study.get_trials(deepcopy=True))
        if frac is not None and len(study.trials) > 2:
            param_space = shrink_param_space(study, param_space, top_frac=frac)
            print(f"[LGBM HPO] Parameter space shrunk: {param_space}")
            logger.info(f"Parameter space shrunk to: {param_space}")

    # === Output best found parameters and performance ===
    print("\n[LGBM HPO] === Best results from all phases ===")
    logger.info("==== Best results from all phases ====")
    print(f"[LGBM HPO] Best R2: {all_study.best_value:.5f}")
    logger.info(f"Best R2: {all_study.best_value:.5f}")
    print("[LGBM HPO] Best Params:")
    for k, v in all_study.best_params.items():
        print(f"    {k}: {v}")
        logger.info(f"  {k}: {v}")

    out = {
        "best_score": all_study.best_value,
        "best_params": all_study.best_params,
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/lgbm_best_params.json", "w") as f:
        json.dump(out, f, indent=2)
    print("[LGBM HPO] Done! Best parameters saved to results/lgbm_best_params.json.")

if __name__ == "__main__":
    # === Run staged HPO process ===
    print("\n[LGBM HPO] Starting LightGBM hyperparameter optimization...")
    staged_hpo([
        # (phase name, seconds, top-fraction for shrinking)
        ("Phase 1", 80*60, 0.3),   # 80 min, shrink to top 30%
        ("Phase 2", 30*60, 0.05),   # 30 min, shrink to top 5%
        ("Phase 3", 10*60, None)    # 10 min, no further shrink
    ], timeout=2*60*60)  # Total timeout: 2 hours
    print("[LGBM HPO] LightGBM HPO complete.")

    # === FINAL OOF PREDICTIONS WITH BEST PARAMETERS ===
    print("\n[LGBM HPO] Generating final OOF predictions with best parameters...")
    DATA_DIR = Path("../data/exam_dataset/1")
    OOF_DIR = Path("results/oof")
    X = pd.read_parquet(DATA_DIR / "X_train.parquet")
    y = pd.read_parquet(DATA_DIR / "y_train.parquet")["SALE_PRC"]
    cat_cols = list(X.select_dtypes(include=["category"]).columns)
    oof_pred = np.zeros(len(y))
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Load best parameters found during HPO
    with open("results/lgbm_best_params.json", "r") as f:
        best_dict = json.load(f)
    best_params = best_dict["best_params"]

    # Standard CV loop to generate OOF predictions for ensembling
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        print(f"[LGBM HPO] Fold {fold}")
        X_train = X.iloc[tr_idx].copy()
        X_valid = X.iloc[va_idx].copy()
        for col in cat_cols:
            X_train[col] = X_train[col].astype("category")
            X_valid[col] = X_valid[col].astype("category")
        model = LGBMRegressor(**best_params, random_state=42)
        model.fit(
            X_train, y.iloc[tr_idx],
            eval_set=[(X_valid, y.iloc[va_idx])],
            callbacks=[early_stopping(stopping_rounds=75, verbose=False)],
            categorical_feature=cat_cols if cat_cols else "auto"
        )
        pred = model.predict(X_valid)
        oof_pred[va_idx] = pred
        score = r2_score(y.iloc[va_idx], pred)
        print(f"  R2 = {score:.5f}")

    # Store OOF predictions to disk for use in stacking/ensembling
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    df_oof = pd.DataFrame({"oof_lgbm_hpo": oof_pred}, index=X.index)
    df_oof.to_parquet(OOF_DIR / "oof_lgbm_hpo.parquet")
    print(f"[LGBM HPO] OOF predictions saved to: {OOF_DIR / 'oof_lgbm_hpo.parquet'}")