import logging
from typing import Optional
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


class AutoML:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model: Optional[Pipeline] = None

    def fit(self, X, y):
        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ]
        )

        models_with_params = {
            "RandomForest": (
                RandomForestRegressor(random_state=self.seed),
                {
                    "model__n_estimators": [50, 100, 200],
                    "model__max_depth": [None, 5, 10],
                }
            ),
            "GradientBoosting": (
                GradientBoostingRegressor(random_state=self.seed),
                {
                    "model__n_estimators": [100, 200],
                    "model__learning_rate": [0.05, 0.1],
                    "model__max_depth": [3, 5],
                }
            ),
            "Ridge": (
                Ridge(),
                {
                    "model__alpha": [0.1, 1.0, 10.0, 100.0],
                }
            ),
            "XGBoost": (
                XGBRegressor(random_state=self.seed, verbosity=0, n_jobs=-1),
                {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [3, 5],
                    "model__learning_rate": [0.05, 0.1, 0.2],
                }
            )
        }

        best_score = -np.inf
        best_model = None

        for name, (model, param_dist) in models_with_params.items():
            logger.info(f"Optimizing {name}...")

            pipeline = Pipeline([
                ("preprocessing", preprocessor),
                ("model", model)
            ])

            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_dist,
                n_iter=9,
                cv=3,
                scoring="r2",
                n_jobs=-1,
                random_state=self.seed,
                verbose=1
            )
            search.fit(X, y)
            score = search.best_score_
            logger.info(f"{name} best R²: {score:.4f}")

            if score > best_score:
                best_score = score
                best_model = search.best_estimator_

        self.model = best_model
        logger.info("Best model selected and trained.")

    def predict(self, X):
        return self.model.predict(X)
