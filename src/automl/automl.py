import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_col="date"):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.date_col in X.columns:
            X[self.date_col] = pd.to_datetime(X[self.date_col])
            X["day_of_week"] = X[self.date_col].dt.dayofweek
            X["month"] = X[self.date_col].dt.month
            X = X.drop(columns=[self.date_col])
        return X

class AutoML:
    def __init__(self, seed=42, n_iter=20, cv=5):
        self.seed = seed
        self.n_iter = n_iter
        self.cv = cv
        self.best_pipeline = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Feature engineering: Datumsextraktion
        X = DateFeatureExtractor().fit_transform(X)

        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ])

        rf = RandomForestRegressor(random_state=self.seed)
        lgbm = lgb.LGBMRegressor(random_state=self.seed)

        ensemble = VotingRegressor([("rf", rf), ("lgbm", lgbm)])

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", ensemble)
        ])

        param_distributions = {
            "model__rf__n_estimators": [50, 100, 200],
            "model__rf__max_depth": [None, 10, 20],
            "model__lgbm__n_estimators": [50, 100, 200],
            "model__lgbm__max_depth": [-1, 10, 20],
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
        print(f"Best params: {search.best_params_}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.best_pipeline is None:
            raise RuntimeError("You must fit the model before prediction.")
        X = DateFeatureExtractor().fit_transform(X)
        return self.best_pipeline.predict(X)
