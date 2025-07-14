from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd
import lightgbm as lgb

class AutoML:
    def __init__(self, seed=42, n_iter=20, cv=3):
        self.seed = seed
        self.n_iter = n_iter
        self.cv = cv
        self.model = None
        self.best_pipeline = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Spalten nach Typ trennen
        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # Vorverarbeitung für numerische Daten
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        # Vorverarbeitung für kategorische Daten
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        # Gesamte Vorverarbeitung
        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ])

        # Modelle definieren
        rf = RandomForestRegressor(random_state=self.seed)
        lgbm = lgb.LGBMRegressor(random_state=self.seed)

        # Ensemble der Modelle
        ensemble = VotingRegressor([("rf", rf), ("lgbm", lgbm)])

        # Pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", ensemble)
        ])

        # Hyperparameter-Suchraum
        param_distributions = {
            "model__rf__n_estimators": [50, 100, 200],
            "model__rf__max_depth": [None, 10, 20, 30],
            "model__lgbm__n_estimators": [50, 100, 200],
            "model__lgbm__max_depth": [-1, 10, 20, 30],
            "model__lgbm__learning_rate": [0.01, 0.05, 0.1]
        }

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_distributions,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring="r2",
            random_state=self.seed,
            n_jobs=-1,
            verbose=1
        )

        search.fit(X, y)
        self.best_pipeline = search.best_estimator_
        print(f"Best parameters: {search.best_params_}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.best_pipeline is None:
            raise RuntimeError("Fit the model first before predicting!")
        return self.best_pipeline.predict(X)
