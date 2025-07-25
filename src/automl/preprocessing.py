# src/automl/preprocessing.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA



class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor class to handle missing values, encoding of categorical features, and optional scaling.
    """

    def __init__(self, use_pca: bool = False, n_components: float = 0.95):
        self.use_pca = use_pca
        self.n_components = n_components
        self.pipeline = None
        self.meta_features_ = None

    def extract_meta_features(self, X, y=None):
        """Extracts dataset-level meta-features."""
        meta_features = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_numeric": X.select_dtypes(include=["number"]).shape[1],
            "n_categorical": X.select_dtypes(exclude=["number"]).shape[1],
            "missing_values": X.isnull().sum().sum(),
        }
        if y is not None:
            meta_features.update({
                "y_mean": y.mean(),
                "y_std": y.std(),
                "y_skew": y.skew(),
            })
        self.meta_features_ = meta_features

    def fit(self, X: pd.DataFrame, y=None):
        # Separate numerical and categorical features
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        # Define transformations
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

        if self.scale_numeric:
            numeric_transformer.steps.append(("scaler", StandardScaler()))

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
        ])

        # Combine into full pipeline
        self.pipeline = ColumnTransformer(transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ])

        self.pipeline.fit(X)

        # Get feature names after encoding (optional)
        cat_feature_names = self.pipeline.named_transformers_['cat']\
            .named_steps['encoder'].get_feature_names_out(categorical_features) if categorical_features else []

        self.feature_names = numeric_features + list(cat_feature_names)

        return self

    def transform(self, X):
        return self.pipeline.transform(X)

    def fit_transform(self, X: pd.DataFrame):
        self.meta_features_ = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_categorical": X.select_dtypes(include="category").shape[1],
            "n_numerical": X.select_dtypes(include=["number"]).shape[1],
        }

        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ])

        steps = [("preprocessor", preprocessor)]

        if self.use_pca:
            steps.append(("pca", PCA(n_components=self.n_components)))

        self.pipeline = Pipeline(steps)
        return self.pipeline.fit_transform(X)

    def get_feature_names(self):
        return self.feature_names
