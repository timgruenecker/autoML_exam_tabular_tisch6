# src/automl/preprocessing.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor class to handle missing values, encoding of categorical features, and optional scaling.
    """

    def __init__(self, scale_numeric: bool = True):
        self.scale_numeric = scale_numeric
        self.pipeline = None
        self.feature_names = None

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

    def transform(self, X: pd.DataFrame):
        return self.pipeline.transform(X)

    def fit_transform(self, X: pd.DataFrame, y=None):
        return self.fit(X).transform(X)

    def get_feature_names(self):
        return self.feature_names
