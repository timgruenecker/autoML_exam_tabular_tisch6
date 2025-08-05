import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
import json
import logging
import sys
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
from sklearn.preprocessing import StandardScaler

# === CONSTANTS & LOGGING ===
DATA_DIR = Path("../data/exam_dataset/1")
RESULTS_DIR = Path("results")
OOF_DIR = RESULTS_DIR / "oof"
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "tabnet_kd_hpo.log"

DURATION_IN_MINUTES = 180 # 3h

# Configure logging to file and stdout
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger()
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(logging.StreamHandler(sys.stdout))

def save_json(data, path: Path):
    """Save a dictionary as JSON to the given path, ensuring the directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# === HELPER: Find best GBDT model and associated OOF preds ===
def get_best_gbdt_info():
    """
    Find the best GBDT model by looking for the *_best_params.json in the results folder.
    Returns model name, params file, and the corresponding OOF file.
    Assumes only one best params file exists (as per pipeline design).
    """
    best_param_files = list(RESULTS_DIR.glob("*_best_params.json"))
    if len(best_param_files) != 1:
        raise RuntimeError(f"Expected exactly one *_best_params.json, found: {best_param_files}")
    params_file = best_param_files[0]
    modelname = params_file.stem.replace("_best_params", "").lower()
    oof_file = OOF_DIR / f"oof_{modelname}_final.parquet"
    if not oof_file.exists():
        oof_file = OOF_DIR / f"oof_{modelname}_hpo.parquet"
        if not oof_file.exists():
            oof_file = OOF_DIR / f"oof_{modelname}.parquet"
    if not oof_file.exists():
        raise FileNotFoundError(f"OOF file for best model {modelname} not found!")
    return modelname, params_file, oof_file

# === LOAD DATA & KNOWLEDGE DISTILLATION TARGETS ===
def load_data_and_kd():
    """
    Load training data, find best GBDT OOF predictions, and compute the knowledge distillation targets.
    Returns standardized features, ground truth, and distillation targets.
    """
    print("[TabNet KD] Loading training data...")
    X = pd.read_parquet(DATA_DIR / "X_train.parquet")
    y = pd.read_parquet(DATA_DIR / "y_train.parquet")["SALE_PRC"].values

    # Use the best GBDT model's OOF predictions as soft targets for knowledge distillation
    modelname, params_file, oof_file = get_best_gbdt_info()
    print(f"[TabNet KD] Using OOF preds from: {oof_file} ({modelname})")
    oof_preds = pd.read_parquet(oof_file).iloc[:,0].values

    print(f"[TabNet KD] Loading best hyperparameters from: {params_file}")
    with open(params_file) as f:
        best_params = json.load(f)["best_params"]

    # Compute log1p targets for knowledge distillation (alpha = ground truth, 1-alpha = teacher/GBDT)
    y_log = np.log1p(y).reshape(-1, 1)
    oof_preds_log = np.log1p(oof_preds).reshape(-1, 1)
    alpha = 0.3
    y_distill = (alpha * y_log + (1 - alpha) * oof_preds_log)

    # Prepare features: standardize continuous, encode categoricals as codes
    cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    cont_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    X_tab = X.copy()
    X_tab[cont_cols] = StandardScaler().fit_transform(X_tab[cont_cols])
    for c in cat_cols:
        X_tab[c] = X_tab[c].astype("category").cat.codes
    X_tab = X_tab.values.astype(np.float32)
    return X_tab, y, y_distill

# === OPTUNA OBJECTIVE FOR TABNET-KD ===
def make_objective(X, y, y_distill):
    """
    Returns an Optuna objective that trains TabNet with KD targets and reports mean R2 across folds.
    """
    def objective(trial):
        params = {
            "n_d": trial.suggest_int("n_d", 32, 56, step=8),
            "n_a": trial.suggest_int("n_a", 32, 56, step=8),
            "n_steps": 3,
            "gamma": trial.suggest_float("gamma", 1.0, 1.2),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-7, 1e-5, log=True),
            "lr": trial.suggest_float("lr", 2e-3, 4e-3, log=True),
            "mask_type": "sparsemax",
            "batch_size": 512,
            "virtual_batch_size": 128,
            "n_shared": 1,
            "n_independent": 2,
            "max_epochs": trial.suggest_int("max_epochs", 140, 180, step=20), # Use low epoch count for training, assuming more epochs in the final 5Fold OOF-CV will have a better performance
            "patience": trial.suggest_int("patience", 20, 30, step=5),
        }
        logger.info(f"Trial {trial.number} params: {params}")
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        # 3-fold CV for TabNet HPO (for efficiency)
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X[train_idx], X[valid_idx]
            y_tr, y_val = y_distill[train_idx], y_distill[valid_idx]
            model = TabNetRegressor(
                n_d=params["n_d"], n_a=params["n_a"], n_steps=params["n_steps"],
                gamma=params["gamma"], lambda_sparse=params["lambda_sparse"],
                optimizer_fn=torch.optim.Adam,
                optimizer_params={"lr": params["lr"]},
                mask_type=params["mask_type"],
                n_shared=params["n_shared"], n_independent=params["n_independent"],
                device_name="cuda" if torch.cuda.is_available() else "cpu", verbose=0
            )
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_name=["val"],
                eval_metric=["rmse"],
                max_epochs=params["max_epochs"],
                patience=params["patience"],
                batch_size=params["batch_size"],
                virtual_batch_size=params["virtual_batch_size"],
                num_workers=0,
                drop_last=False,
            )
            # Predict on validation set, invert log1p
            preds_log = model.predict(X_val).reshape(-1)
            preds = np.expm1(preds_log)
            score = r2_score(y[valid_idx], preds)  # Evaluate on true y (not KD)
            scores.append(score)
            trial.report(score, fold)
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at fold {fold}")
                raise optuna.TrialPruned()
        mean_score = np.mean(scores)
        logger.info(f"Trial {trial.number} mean R2 = {mean_score:.5f}")
        return mean_score
    return objective

# === MAIN: RUN HPO, SAVE BEST PARAMS AND FINAL OOF PREDICTIONS ===
if __name__ == "__main__":
    print("[TabNet KD HPO] Starting TabNet-KD hyperparameter optimization...")
    X_tab, y, y_distill = load_data_and_kd()

    # Run Optuna HPO (1 minute timeout for demo; extend for real tuning)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=HyperbandPruner()
    )
    study.optimize(make_objective(X_tab, y, y_distill), timeout=DURATION_IN_MINUTES*60)

    # Output best results
    print("\n[TabNet KD HPO] === Best results ===")
    print(f"[TabNet KD HPO] Best mean R2: {study.best_value:.5f}")
    print("[TabNet KD HPO] Best Params:")
    for k, v in study.best_params.items():
        print(f"   {k}: {v}")
    out = {"best_score": study.best_value, "best_params": study.best_params}
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "tabnet_kd_best_hyperparams.json", "w") as f:
        json.dump(out, f, indent=2)
    print("[TabNet KD HPO] Best KD parameters saved to results/tabnet_kd_best_hyperparams.json")

    # === FINAL OOF PREDICTIONS WITH BEST PARAMETERS ===
    print("\n[TabNet KD HPO] Generating final OOF predictions with best parameters...")
    oof_pred = np.zeros(len(y))
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr, va) in enumerate(cv.split(X_tab)):
        print(f"[TabNet KD Final OOF] Fold {fold}")
        tabnet_init_keys = [
            "n_d", "n_a", "n_steps", "gamma", "lambda_sparse",
            "mask_type", "n_shared", "n_independent"
        ]
        tabnet_init_params = {k: v for k, v in study.best_params.items() if k in tabnet_init_keys}

        mdl = TabNetRegressor(
            **tabnet_init_params,
            optimizer_fn=torch.optim.Adam,
            optimizer_params={"lr": study.best_params["lr"]},
            device_name="cuda" if torch.cuda.is_available() else "cpu",
            verbose=0
        )
        mdl.fit(
            X_tab[tr], y_distill[tr],
            eval_set=[(X_tab[va], y_distill[va])],
            eval_name=["val"],
            eval_metric=["rmse"],
            max_epochs=350, # Use more epochs during OOF-CV to ensure each fold model fully converges. (maybe needs further adjustment when testing on huge datasets)
            patience=150,
            batch_size=512,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False,
        )
        preds_log = mdl.predict(X_tab[va]).reshape(-1)
        preds = np.expm1(preds_log)
        oof_pred[va] = preds

    global_r2 = r2_score(y, oof_pred)
    print(f"[TabNet KD HPO] Final global OOF R2: {global_r2:.5f}")

    OOF_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"oof_tabnet_kd_hpo": oof_pred}, index=np.arange(len(y))).to_parquet(OOF_DIR / "oof_tabnet_kd_hpo.parquet")
    print(f"[TabNet KD HPO] OOF predictions saved to {OOF_DIR}/oof_tabnet_kd_hpo.parquet")