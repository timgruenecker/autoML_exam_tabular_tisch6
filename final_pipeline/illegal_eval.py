import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score

# Pfade anpassen, falls nötig
RESULTS_DIR     = Path("results")
SUBMISSION_FILE = RESULTS_DIR / "submission.csv"
Y_TEST_FILE     = Path("../data/exam_dataset/1") / "y_test.parquet"

# Submission einlesen
sub = pd.read_csv(SUBMISSION_FILE)
y_pred = sub["SALE_PRC"].values

# True Targets einlesen
y_true = pd.read_parquet(Y_TEST_FILE)["SALE_PRC"].values

# R² berechnen
score = r2_score(y_true, y_pred)
print(f"R² between submission and y_test: {score:.5f}")
