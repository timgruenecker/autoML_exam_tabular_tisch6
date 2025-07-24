import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.validation import check_is_fitted
import logging

logger = logging.getLogger(__name__)


class FeatureStacker(BaseEstimator, TransformerMixin):
    """Stacks original and engineered features."""
    def __init__(self, transformers):
        self.transformers = transformers

    def fit(self, X, y=None):
        self.fitted_transformers_ = []
        for name, transformer in self.transformers:
            fitted = transformer.fit(X, y)
            self.fitted_transformers_.append((name, fitted))
        return self

    def transform(self, X):
        features = [t.transform(X) for _, t in self.fitted_transformers_]
        return np.hstack(features)


def build_preprocessing_pipeline(X: pd.DataFrame) -> Pipeline:
    """Constructs the preprocessing pipeline with imputation, encoding, scaling, and feature engineering."""
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()

    logger.info(f"Identified {len(numeric_features)} numeric and {len(categorical_features)} categorical features.")

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    engineered_features = Pipeline([
        ("selector", ColumnTransformer([
            ("numeric", "passthrough", numeric_features)
        ])),
        ("imputer", SimpleImputer(strategy="median")),
        ("poly", PolynomialFeatures(degree=2, include_bias=False))
    ])

    full_pipeline = Pipeline([
        ("union", FeatureStacker([
            ("original", preprocessor),
            ("engineered", engineered_features)
        ])),
        ("var_thresh", VarianceThreshold(threshold=0.0))
    ])

    return full_pipeline
