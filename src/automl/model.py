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
import time

class AutoML:
    def __init__(self, random_state=42):
        # Initialize AutoML with default random state and placeholders
        self.random_state = random_state
        self.pipeline = None
        self.best_model = None
        self.best_score_ = None
        self.training_time = None
        self.memory_usage = None

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        # Detect numeric and categorical columns from input DataFrame
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Numeric pipeline: impute missing values with median, then scale features
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Categorical pipeline: impute missing with constant, then one-hot encode
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine numeric and categorical pipelines using ColumnTransformer
        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

        return preprocessor

    def _build_model(self) -> StackingRegressor:
        # Define base regressors for stacking ensemble
        base_models = [
            ('ridge', Ridge(random_state=self.random_state)),
            ('knn', KNeighborsRegressor()),
            ('xgb', XGBRegressor(objective='reg:squarederror', random_state=self.random_state, verbosity=0))
        ]
        # Define meta-model (final estimator) as Ridge regression
        meta_model = Ridge(random_state=self.random_state)
        # Create stacking regressor with 5-fold CV, parallel jobs, and passthrough of original features
        return StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1,
            passthrough=True
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        logging.info("Starting AutoML fit procedure...")

        start_time = time.time()
        # Build preprocessing pipeline based on input features
        preprocessor = self._build_preprocessor(X)
        # Build the stacking model
        model = self._build_model()

        # Combine preprocessing and model into one pipeline
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        # Define hyperparameter search space for final estimator alpha
        param_distributions = {
            'regressor__final_estimator__alpha': [0.1, 1.0, 10.0],
        }

        # Use randomized search with 3-fold CV to find best hyperparameters
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

        # Fit the model with hyperparameter search
        search.fit(X, y)

        # Record training duration
        self.training_time = time.time() - start_time
        # Estimate memory usage after training
        self.memory_usage = self._estimate_memory_usage()

        # Store best estimator and its CV score
        self.best_model = search.best_estimator_
        self.best_score_ = search.best_score_

        # Log relevant information about training and best parameters
        logging.info(f"Best parameters found: {search.best_params_}")
        logging.info(f"Best CV R² score: {self.best_score_:.4f}")
        logging.info(f"Training time (s): {self.training_time:.2f}")
        logging.info(f"Estimated memory usage (MB): {self.memory_usage:.2f}")

    def _estimate_memory_usage(self) -> float:
        # Estimate current process memory usage in MB using psutil
        import psutil
        process = psutil.Process()
        mem_bytes = process.memory_info().rss
        return mem_bytes / (1024 * 1024)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Predict using the trained best model
        if self.best_model is None:
            raise ValueError("Model has not been trained yet.")
        return self.best_model.predict(X)

    def save(self, filepath: str = 'automl_model.joblib'):
        # Save the trained best model to disk
        if self.best_model is None:
            raise ValueError("No model to save. Train the model first.")
        joblib.dump(self.best_model, filepath)
        logging.info(f"Model saved to {filepath}")

    def load(self, filepath: str = 'automl_model.joblib'):
        # Load a trained model from disk
        self.best_model = joblib.load(filepath)
        logging.info(f"Model loaded from {filepath}")
