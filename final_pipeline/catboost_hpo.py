import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
import numpy as np
import sys, logging, time, json, os

# === CONSTANTS & LOGGING SETUP ===
DATA_DIR = Path("../data/exam_dataset/1")
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "catboost_hpo_final_pipeline.log"

# --- Logging: File + stdout ---
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger()
# Add a single stdout StreamHandler if not already present
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(logging.StreamHandler(sys.stdout))

# === DATA LOADING ===
print("[CatBoost HPO] Loading training data...")
X = pd.read_parquet(DATA_DIR / "X_train.parquet")
y = pd.read_parquet(DATA_DIR / "y_train.parquet")["SALE_PRC"]
categorical_cols = list(X.select_dtypes(include=["category"]).columns)
print(f"[CatBoost HPO] Data loaded. n_samples={len(X)}, n_features={X.shape[1]}, n_cats={len(categorical_cols)}")

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# === PARAMETER SPACE FOR HPO ===
def get_param_space(bounds=None):
    """Return the full hyperparameter search space for CatBoost."""
    if bounds is None:
        bounds = {}
    space = {
        "iterations": ("int", bounds.get("iterations", (1500, 3500))),
        "learning_rate": ("float", bounds.get("learning_rate", (0.01, 0.07))),
        "depth": ("int", bounds.get("depth", (4, 8))),
        "l2_leaf_reg": ("float", bounds.get("l2_leaf_reg", (1e-7, 1e-2))),
        "bagging_temperature": ("float", bounds.get("bagging_temperature", (0.05, 0.4))),
        "random_strength": ("float", bounds.get("random_strength", (1.0, 5.0)))
    }
    return space

def make_objective(param_space):
    """
    Returns an Optuna objective function for CatBoost HPO.
    Uses cross-validation (5 folds) and returns the mean R2.
    """
    def objective(trial):
        params = {}
        for key, (typ, val) in param_space.items():
            if typ == "int":
                params[key] = trial.suggest_int(key, val[0], val[1])
            elif typ == "float":
                if key in ["learning_rate", "l2_leaf_reg"]:
                    params[key] = trial.suggest_float(key, val[0], val[1], log=True)
                else:
                    params[key] = trial.suggest_float(key, val[0], val[1])
            elif typ == "cat":
                params[key] = trial.suggest_categorical(key, val)
        params.update({
            "random_seed": 42,
            "logging_level": "Silent",
            "task_type": "CPU",
            "early_stopping_rounds": 75,
            "thread_count": 2,
        })

        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model = CatBoostRegressor(**params)
            model.fit(
                X_train, y_train,
                cat_features=categorical_cols,
                eval_set=(X_val, y_val),
                early_stopping_rounds=75,
                use_best_model=True
            )
            preds = model.predict(X_val)
            score = r2_score(y_val, preds)
            scores.append(score)
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
    Reduce the hyperparameter space to focus around the top X% of trials.
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
            if lo == hi:
                hi = lo + (1 if typ == "int" else 1e-4)
            new_space[key] = (typ, (type(val[0])(lo), type(val[1])(hi)))
    return new_space

# === PHASED, TIME-BASED HPO ===
def staged_hpo(
    phases=[("Phase 1", 60*60, 0.30), ("Phase 2", 48*60, 0.12), ("Phase 3", 12*60, None)],
    timeout=2*60*60
):
    """
    Run staged Optuna HPO: Phase 1 (broad), Phase 2/3 (narrowed).
    Each phase can shrink the search space to the top X% of previous results.
    """
    all_study = None
    param_space = get_param_space(bounds={})
    t0 = time.time()
    for pname, sec, frac in phases:
        print(f"\n[CatBoost HPO] === {pname}: {sec//60} min ===")
        logger.info(f"===== {pname}: {sec//60} min =====")
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=HyperbandPruner()
        )
        time_left = timeout - (time.time() - t0)
        phase_time = min(sec, time_left)
        if phase_time <= 0:
            print("[CatBoost HPO] Time budget exceeded, stopping.")
            break
        print(f"[CatBoost HPO] Optimizing for up to {phase_time/60:.1f} min...")
        study.optimize(make_objective(param_space), timeout=phase_time, n_jobs=-1)
        if all_study is None:
            all_study = study
        else:
            all_study.add_trials(study.get_trials(deepcopy=True))
        if frac is not None and len(study.trials) > 2:
            param_space = shrink_param_space(study, param_space, top_frac=frac)
            print(f"[CatBoost HPO] Reduced search space: {param_space}")
            logger.info(f"Parameter space shrunk to: {param_space}")

    print("\n[CatBoost HPO] === Best results from all phases ===")
    logger.info("==== Best results from all phases ====")
    print(f"[CatBoost HPO] Best R2: {all_study.best_value:.5f}")
    logger.info(f"Best R2: {all_study.best_value:.5f}")
    print("[CatBoost HPO] Best Params:")
    for k, v in all_study.best_params.items():
        print(f"    {k}: {v}")
        logger.info(f"  {k}: {v}")

    # Save best params and score for downstream scripts
    out = {
        "best_score": all_study.best_value,
        "best_params": all_study.best_params,
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/catboost_best_params.json", "w") as f:
        json.dump(out, f, indent=2)
    print("[CatBoost HPO] Done! Best parameters saved to results/catboost_best_params.json.")

if __name__ == "__main__":
    print("\n[CatBoost HPO] Starting CatBoost hyperparameter optimization...")
    staged_hpo([
        # (phase name, seconds, top-fraction for shrinking)
        ("Phase 1", 80*60, 0.3),   # 80 min, shrink to top 30%
        ("Phase 2", 30*60, 0.05),   # 30 min, shrink to top 5%
        ("Phase 3", 10*60, None)    # 10 min, no further shrink
    ], timeout=2*60*60)  # Total timeout: 2 hours
    print("[CatBoost HPO] CatBoost HPO complete.")

    # === OOF-PREDICTIONS USING BEST PARAMS FROM HPO ===
    print("\n[CatBoost HPO] Generating final OOF predictions with best parameters...")
    DATA_DIR = Path("../data/exam_dataset/1")
    OOF_DIR = Path("results/oof")
    X = pd.read_parquet(DATA_DIR / "X_train.parquet")
    y = pd.read_parquet(DATA_DIR / "y_train.parquet")["SALE_PRC"]
    cat_cols = list(X.select_dtypes(include=["category"]).columns)
    oof_pred = np.zeros(len(y))
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Load best parameters
    with open("results/catboost_best_params.json", "r") as f:
        best_dict = json.load(f)
    best_params = best_dict["best_params"]

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        print(f"[CatBoost HPO] Fold {fold}")
        model = CatBoostRegressor(**best_params, verbose=0, train_dir="catboost_tmp", random_seed=42, task_type="CPU")
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            cat_features=cat_cols,
            eval_set=(X.iloc[va_idx], y.iloc[va_idx]),
            early_stopping_rounds=75,
            use_best_model=True
        )
        oof_pred[va_idx] = model.predict(X.iloc[va_idx])

    # Calculate OOF R2 score for all data
    oof_score = r2_score(y, oof_pred)
    print(f"[CatBoost HPO] Final OOF R2 across all folds: {oof_score:.5f}")

    OOF_DIR.mkdir(parents=True, exist_ok=True)
    df_oof = pd.DataFrame({"oof_catboost_hpo": oof_pred}, index=X.index)
    df_oof.to_parquet(OOF_DIR / "oof_catboost_hpo.parquet")
    print(f"[CatBoost HPO] OOF predictions saved to: {OOF_DIR / 'oof_catboost_hpo.parquet'}")
