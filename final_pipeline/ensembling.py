import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
import json

# ——— Paths and model setup ————————————————————————————————————————————————
DATA_DIR             = Path("../data/exam_dataset/1")
OOF_DIR              = Path("results/oof")
RESULTS_DIR          = Path("results")
ENSEMBLE_RESULT_FILE = RESULTS_DIR / "ensemble_eval.json"
SUBMISSION_FILE      = RESULTS_DIR / "submission.csv"

MODEL_NAMES = [
    "catboost", "lightgbm", "xgboost",
    "tabnet_kd_hpo", "linearregression", "randomforest"
]

def load_best_params(model_name):
    """Load best parameters for a given model from JSON file."""
    param_file = RESULTS_DIR / f"{model_name}_best_params.json"
    if param_file.exists():
        with open(param_file, "r") as f:
            params = json.load(f)
            return params.get("best_params", params)
    return {}

# ——— Load training and test data ——————————————————————————————————————————————
X_train_full = pd.read_parquet(DATA_DIR / 'X_train.parquet')
y_train_full = pd.read_parquet(DATA_DIR / 'y_train.parquet')["SALE_PRC"].values
X_test_full  = pd.read_parquet(DATA_DIR / 'X_test.parquet')

cat_cols  = X_train_full.select_dtypes(include=["category","object"]).columns.tolist()
cont_cols = X_train_full.select_dtypes(include=["int64","float64","bool"]).columns.tolist()

# ——— Load OOF predictions for all base models —————————————————————————————————
oof_preds_list = []
for name in MODEL_NAMES:
    fn = OOF_DIR / f"oof_{name}.parquet"
    if not fn.exists():
        fn = OOF_DIR / f"oof_{name}_final.parquet"
    df = pd.read_parquet(fn)
    col = [c for c in df.columns if c.startswith("oof_")][0]
    oof_preds_list.append(df[col].values)
oof_preds = np.column_stack(oof_preds_list)

# ——— Find the best baseline (single model) using OOF R² —————————————————————————
best_base_score = -np.inf
best_base_name  = None
for name in MODEL_NAMES:
    fn = OOF_DIR / f"oof_{name}_final.parquet"
    if fn.exists():
        y_oof = pd.read_parquet(fn).iloc[:,0].values
        r2 = r2_score(y_train_full, y_oof)
        if r2 > best_base_score:
            best_base_score = r2
            best_base_name  = name
print(f"[Baseline OOF] Best model: {best_base_name} with R² = {best_base_score:.5f}")

# ——— Evaluate stacking candidates via OOF-CV for all model combinations ————————
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

meta_specs = {
    "Ridge": {
        "class": Ridge,
        "params": dict(alpha=1.0)
    }
}

all_combos = [
    combo
    for r in range(2, len(MODEL_NAMES)+1)
    for combo in combinations(range(len(MODEL_NAMES)), r)
]

results = {meta_name: {} for meta_name in meta_specs}

for combo in all_combos:
    Xm = oof_preds[:, combo]
    names = [MODEL_NAMES[i] for i in combo]
    print(f"OOF-CV testing combo: {names}")
    for meta_name, spec in meta_specs.items():
        meta_oof = np.zeros_like(y_train_full, dtype=float)
        for tr, va in kf.split(Xm):
            m = spec["class"](**spec["params"])
            m.fit(Xm[tr], y_train_full[tr])
            meta_oof[va] = m.predict(Xm[va])
        score = r2_score(y_train_full, meta_oof)
        print(f"  {meta_name} → R² (OOF-CV) = {score:.5f}")
        m_final = spec["class"](**spec["params"])
        m_final.fit(Xm, y_train_full)
        results[meta_name][combo] = {
            "score": score,
            "model": m_final,
            "used_models": names
        }

# ——— Identify best ensemble based on OOF-CV ———————————————————————————————
best_ensemble_score = best_base_score
best_ensemble_info  = None
best_ensemble_model = None

for meta_name, combos in results.items():
    for combo, info in combos.items():
        if info["score"] > best_ensemble_score:
            best_ensemble_score = info["score"]
            best_ensemble_info  = {
                "meta_model": meta_name,
                "models": info["used_models"],
                "combo": combo,
                "score": info["score"]
            }
            best_ensemble_model = info["model"]

print(f"\n[Best Ensemble OOF] {best_ensemble_info}")

# ——— Save summary of results ————————————————————————————————————————————————
RESULTS_DIR.mkdir(exist_ok=True)
with open(ENSEMBLE_RESULT_FILE, "w") as f:
    json.dump({
        "baseline_oof": {"model": best_base_name, "r2": best_base_score},
        "best_ensemble_oof": best_ensemble_info
    }, f, indent=2)

# ——— Helper function to fit and predict base models (incl. correct TabNet scaling) —— 
def get_base_model_preds(names):
    train_preds = []
    test_preds  = []
    for name in names:
        params = load_best_params(name)
        if name == "catboost":
            m = CatBoostRegressor(**params, verbose=0, train_dir="catboost_tmp", random_seed=42)
            Xtr, Xte = X_train_full.copy(), X_test_full.copy()
            for c in cat_cols:
                Xtr[c], Xte[c] = Xtr[c].astype(str), Xte[c].astype(str)
            m.fit(Xtr, y_train_full, cat_features=cat_cols)
            train_preds.append(m.predict(Xtr))
            test_preds.append(m.predict(Xte))

        elif name == "lightgbm":
            m = LGBMRegressor(**params, random_state=42)
            Xtr, Xte = X_train_full.copy(), X_test_full.copy()
            for c in cat_cols:
                Xtr[c], Xte[c] = Xtr[c].astype("category").cat.codes, Xte[c].astype("category").cat.codes
            m.fit(Xtr, y_train_full)
            train_preds.append(m.predict(Xtr))
            test_preds.append(m.predict(Xte))

        elif name == "xgboost":
            m = XGBRegressor(**params, random_state=42)
            Xtr, Xte = X_train_full.copy(), X_test_full.copy()
            for c in cat_cols:
                Xtr[c], Xte[c] = Xtr[c].astype("category").cat.codes, Xte[c].astype("category").cat.codes
            m.fit(Xtr, y_train_full)
            train_preds.append(m.predict(Xtr))
            test_preds.append(m.predict(Xte))

        elif name == "randomforest":
            m = RandomForestRegressor(**params, random_state=42)
            Xtr, Xte = X_train_full.copy(), X_test_full.copy()
            for c in cat_cols:
                Xtr[c], Xte[c] = Xtr[c].astype("category").cat.codes, Xte[c].astype("category").cat.codes
            m.fit(Xtr, y_train_full)
            train_preds.append(m.predict(Xtr))
            test_preds.append(m.predict(Xte))

        elif name == "linearregression":
            m = LinearRegression()
            Xtr, Xte = X_train_full[cont_cols], X_test_full[cont_cols]
            m.fit(Xtr, y_train_full)
            train_preds.append(m.predict(Xtr))
            test_preds.append(m.predict(Xte))

        elif name == "tabnet_kd_hpo":
            with open(RESULTS_DIR / "tabnet_kd_best_hyperparams.json") as f:
                bp = json.load(f)["best_params"]
            oof_best = pd.read_parquet(OOF_DIR / f"oof_{best_base_name}_final.parquet").iloc[:,0].values
            y_log   = np.log1p(y_train_full).reshape(-1,1)
            oof_log = np.log1p(oof_best).reshape(-1,1)
            y_dist  = 0.3 * y_log + 0.7 * oof_log

            scaler_tab = StandardScaler()
            Xtr, Xte = X_train_full.copy(), X_test_full.copy()
            Xtr[cont_cols] = scaler_tab.fit_transform(Xtr[cont_cols])
            Xte[cont_cols] = scaler_tab.transform(Xte[cont_cols])
            for c in cat_cols:
                Xtr[c], Xte[c] = Xtr[c].astype("category").cat.codes, Xte[c].astype("category").cat.codes

            Xtr_np, Xte_np = Xtr.values.astype(np.float32), Xte.values.astype(np.float32)
            tabnet = TabNetRegressor(
                n_d=int(bp["n_d"]), n_a=int(bp["n_a"]), n_steps=3,
                gamma=float(bp["gamma"]), lambda_sparse=float(bp["lambda_sparse"]),
                optimizer_fn=torch.optim.Adam,
                optimizer_params={"lr": float(bp["lr"])},
                mask_type="sparsemax",
                n_shared=1, n_independent=2,
                seed=42, verbose=0,
                device_name="cuda" if torch.cuda.is_available() else "cpu"
            )
            tabnet.fit(
                Xtr_np, y_dist,
                eval_set=[(Xtr_np, y_dist)], eval_name=["train"],
                eval_metric=["rmse"],
                max_epochs=180, patience=150, # 350 epochs hardcoded, as it was a good avg for all datasets (maybe needs further adjustment when testing on huge datasets)
                batch_size=512, virtual_batch_size=128,
                num_workers=0, drop_last=False
            )
            train_preds.append(np.expm1(tabnet.predict(Xtr_np).reshape(-1)))
            test_preds.append(np.expm1(tabnet.predict(Xte_np).reshape(-1)))

        else:
            raise ValueError(f"Unknown model: {name}")

    return np.column_stack(train_preds), np.column_stack(test_preds)

# ——— Fit all base models once and predict on train/test for ensembling ————————
_, test_preds_all = get_base_model_preds(MODEL_NAMES)

# ——— Predict and create submission based on best ensemble (or fallback baseline) —— 
if best_ensemble_info is not None:
    combo       = best_ensemble_info["combo"]
    names_str   = ", ".join(best_ensemble_info["models"])
    X_test_meta = test_preds_all[:, combo]
    y_pred_ens  = best_ensemble_model.predict(X_test_meta)
    print(f"\nBest Ensemble on Testset: Combo ({names_str}) → R² OOF-CV = {best_ensemble_score:.5f}")

    submission = pd.DataFrame({
        "Id": np.arange(len(y_pred_ens)),
        "SALE_PRC": y_pred_ens
    })
    submission.to_csv(SUBMISSION_FILE, index=False)
    print(f"[Submission] Saved ensemble predictions to {SUBMISSION_FILE}")

else:
    print(f"\nNo ensemble beat the baseline on OOF; using baseline {best_base_name}.")
    params = load_best_params(best_base_name)
    if best_base_name == "catboost":
        m = CatBoostRegressor(**params, verbose=0, train_dir="catboost_tmp", random_seed=42)
        Xtr, Xte = X_train_full.copy(), X_test_full.copy()
        for c in cat_cols:
            Xtr[c], Xte[c] = Xtr[c].astype(str), Xte[c].astype(str)
        m.fit(Xtr, y_train_full, cat_features=cat_cols)
        y_pred_ens = m.predict(Xte)
    elif best_base_name == "lightgbm":
        m = LGBMRegressor(**params, random_state=42)
        Xtr, Xte = X_train_full.copy(), X_test_full.copy()
        for c in cat_cols:
            Xtr[c], Xte[c] = Xtr[c].astype("category").cat.codes, Xte[c].astype("category").cat.codes
        m.fit(Xtr, y_train_full)
        y_pred_ens = m.predict(Xte)
    elif best_base_name == "xgboost":
        m = XGBRegressor(**params, random_state=42)
        Xtr, Xte = X_train_full.copy(), X_test_full.copy()
        for c in cat_cols:
            Xtr[c], Xte[c] = Xtr[c].astype("category").cat.codes, Xte[c].astype("category").cat.codes
        m.fit(Xtr, y_train_full)
        y_pred_ens = m.predict(Xte)
    elif best_base_name == "randomforest":
        m = RandomForestRegressor(**params, random_state=42)
        Xtr, Xte = X_train_full.copy(), X_test_full.copy()
        for c in cat_cols:
            Xtr[c], Xte[c] = Xtr[c].astype("category").cat.codes, Xte[c].astype("category").cat.codes
        m.fit(Xtr, y_train_full)
        y_pred_ens = m.predict(Xte)
    elif best_base_name == "linearregression":
        m = LinearRegression()
        Xtr, Xte = X_train_full[cont_cols], X_test_full[cont_cols]
        m.fit(Xtr, y_train_full)
        y_pred_ens = m.predict(Xte)
    else:
        raise ValueError(f"Unknown model: {best_base_name}")

    submission = pd.DataFrame({
        "Id": np.arange(len(y_pred_ens)),
        "SALE_PRC": y_pred_ens
    })
    submission.to_csv(SUBMISSION_FILE, index=False)
    print(f"[Submission] Saved baseline predictions to {SUBMISSION_FILE}")

print(f"\nDetails: {ENSEMBLE_RESULT_FILE}")
