import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score

class AutoML:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.pipeline = None
        self.best_model = None
        self.best_score_ = None

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

        return preprocessor

    def _build_model(self) -> StackingRegressor:
        base_models = [
            ('ridge', Ridge(random_state=self.random_state)),
            ('knn', KNeighborsRegressor()),
            ('xgb', XGBRegressor(objective='reg:squarederror', random_state=self.random_state, verbosity=0))
        ]
        meta_model = Ridge(random_state=self.random_state)
        stacking_regressor = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1,
            passthrough=True
        )
        return stacking_regressor

    def fit(self, X: pd.DataFrame, y: pd.Series):
        logging.info("Starting AutoML fit procedure...")
        preprocessor = self._build_preprocessor(X)
        model = self._build_model()

        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        param_distributions = {
            'regressor__final_estimator__alpha': [0.1, 1.0, 10.0],
            'regressor__estimators': [
                [
                    ('ridge', Ridge(alpha=1.0, random_state=self.random_state)),
                    ('knn', KNeighborsRegressor(n_neighbors=5)),
                    ('xgb', XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=self.random_state, verbosity=0))
                ],
                # Du kannst weitere Modelkombinationen hier definieren, falls gewünscht
            ],
        }

        search = RandomizedSearchCV(
            self.pipeline,
            param_distributions=param_distributions,
            n_iter=5,
            cv=3,
            scoring='r2',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=2
        )

        search.fit(X, y)
        self.best_model = search.best_estimator_
        self.best_score_ = search.best_score_

        logging.info(f"Best parameters found: {search.best_params_}")
        logging.info(f"Best CV R² score: {self.best_score_:.4f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("Model has not been trained yet.")
        return self.best_model.predict(X)

    def save(self, filepath: str = 'automl_model.joblib'):
        if self.best_model is None:
            raise ValueError("No model to save. Train the model first.")
        joblib.dump(self.best_model, filepath)
        logging.info(f"Model saved to {filepath}")

    def load(self, filepath: str = 'automl_model.joblib'):
        self.best_model = joblib.load(filepath)
        logging.info(f"Model loaded from {filepath}")
