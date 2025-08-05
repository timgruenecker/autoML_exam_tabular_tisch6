import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
import sys, logging, time, json
import numpy as np
import os

# === CONSTANTS & LOGGING ===
DATA_DIR = Path("../data/exam_dataset/1")
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "rf_hpo_final_pipeline.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger()
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(logging.StreamHandler(sys.stdout))

# === DATA LOADING ===
print("[RF HPO] Loading training data...")
X = pd.read_parquet(DATA_DIR / "X_train.parquet")
y = pd.read_parquet(DATA_DIR / "y_train.parquet")["SALE_PRC"]

# RandomForest does not support categoricals: encode as codes
categorical_cols = list(X.select_dtypes(include=["category"]).columns)
for col in categorical_cols:
    X[col] = X[col].cat.codes

print(f"[RF HPO] Data loaded. n_samples={len(X)}, n_features={X.shape[1]}, n_cats={len(categorical_cols)}")

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# === PARAMETER SPACE ===
def get_param_space(bounds=None):
    """Return the search space for RandomForest hyperparameters."""
    if bounds is None:
        bounds = {}
    space = {
        "n_estimators": ("int", bounds.get("n_estimators", (300, 2000))),
        "max_depth": ("int", bounds.get("max_depth", (3, 20))),
        "max_features": ("float", bounds.get("max_features", (0.4, 1.0))),
        "min_samples_split": ("int", bounds.get("min_samples_split", (2, 20))),
        "min_samples_leaf": ("int", bounds.get("min_samples_leaf", (1, 15))),
        "bootstrap": ("cat", bounds.get("bootstrap", [True, False])),
    }
    return space

def make_objective(param_space):
    """Create an Optuna objective function for RandomForest HPO with cross-validation."""
    def objective(trial):
        params = {}
        for key, (typ, val) in param_space.items():
            if typ == "int":
                params[key] = trial.suggest_int(key, val[0], val[1])
            elif typ == "float":
                params[key] = trial.suggest_float(key, val[0], val[1])
            elif typ == "cat":
                params[key] = trial.suggest_categorical(key, val)
        params.update({
            "n_jobs": 2,
            "random_state": 42,
        })

        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)
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
    """Shrink the parameter space to focus around the top fraction of best trials."""
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

# === STAGED, TIME-BASED HPO ===
def staged_hpo(
    phases=[("Phase 1", 55*60, 0.40), ("Phase 2", 43*60, 0.15), ("Phase 3", 7*60, None)],
    timeout=2*60*60
):
    """
    Run staged Optuna HPO: multiple phases, shrinking parameter space after each phase.
    """
    all_study = None
    param_space = get_param_space(bounds={})
    t0 = time.time()
    for pname, sec, frac in phases:
        print(f"\n[RF HPO] === {pname}: {sec//60} min ===")
        logger.info(f"===== {pname}: {sec//60} min =====")
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=HyperbandPruner()
        )
        time_left = timeout - (time.time() - t0)
        phase_time = min(sec, time_left)
        if phase_time <= 0:
            print("[RF HPO] Time budget exceeded, stopping.")
            break
        print(f"[RF HPO] Optimizing for up to {phase_time/60:.1f} min...")
        study.optimize(make_objective(param_space), timeout=phase_time, n_jobs=-1)
        if all_study is None:
            all_study = study
        else:
            all_study.add_trials(study.get_trials(deepcopy=True))
        if frac is not None and len(study.trials) > 2:
            param_space = shrink_param_space(study, param_space, top_frac=frac)
            print(f"[RF HPO] Reduced parameter space: {param_space}")
            logger.info(f"Parameter space shrunk to: {param_space}")

    print("\n[RF HPO] === Best results from all phases ===")
    logger.info("==== Best results from all phases ====")
    print(f"[RF HPO] Best R2: {all_study.best_value:.5f}")
    logger.info(f"Best R2: {all_study.best_value:.5f}")
    print("[RF HPO] Best Params:")
    for k, v in all_study.best_params.items():
        print(f"    {k}: {v}")
        logger.info(f"  {k}: {v}")

    out = {
        "best_score": all_study.best_value,
        "best_params": all_study.best_params,
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/rf_best_params.json", "w") as f:
        json.dump(out, f, indent=2)
    print("[RF HPO] Done! Best parameters saved to results/rf_best_params.json.")

if __name__ == "__main__":
    print("\n[RF HPO] Starting RandomForest hyperparameter optimization...")
    staged_hpo([
        # (phase name, seconds, top-fraction for shrinking)
        ("Phase 1", 80*60, 0.3),   # 80 min, shrink to top 30%
        ("Phase 2", 30*60, 0.05),   # 30 min, shrink to top 5%
        ("Phase 3", 10*60, None)    # 10 min, no further shrink
    ], timeout=2*60*60)  # Total timeout: 2 hours
    print("[RF HPO] RF HPO complete.")

    # === OOF-PREDICTIONS WITH BEST PARAMS AFTER HPO ===
    print("\n[RF HPO] Generating final OOF predictions with best parameters...")
    DATA_DIR = Path("../data/exam_dataset/1")
    OOF_DIR = Path("results/oof")
    X = pd.read_parquet(DATA_DIR / "X_train.parquet")
    y = pd.read_parquet(DATA_DIR / "y_train.parquet")["SALE_PRC"]

    # Encode categoricals (same as in HPO)
    cat_cols = list(X.select_dtypes(include=["category"]).columns)
    if cat_cols:
        for col in cat_cols:
            X[col] = X[col].cat.codes

    oof_pred = np.zeros(len(y))
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Load best parameters
    with open("results/rf_best_params.json", "r") as f:
        best_dict = json.load(f)
    best_params = best_dict["best_params"]

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        print(f"[RF HPO] Fold {fold}")
        model = RandomForestRegressor(**best_params, random_state=42)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = model.predict(X.iloc[va_idx])
        oof_pred[va_idx] = pred
        score = r2_score(y.iloc[va_idx], pred)
        print(f"  R2 = {score:.5f}")

    OOF_DIR.mkdir(parents=True, exist_ok=True)
    df_oof = pd.DataFrame({"oof_rf_hpo": oof_pred}, index=X.index)
    df_oof.to_parquet(OOF_DIR / "oof_rf_hpo.parquet")
    print(f"[RF HPO] OOF predictions saved to: {OOF_DIR / 'oof_rf_hpo.parquet'}")