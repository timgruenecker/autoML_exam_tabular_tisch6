from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, use_pca=False, pca_variance=0.95):
        """
        Initializes the preprocessing pipeline.

        Args:
            use_pca (bool): Whether to apply PCA dimensionality reduction.
            pca_variance (float): Fraction of variance to retain when using PCA.
        """
        self.use_pca = use_pca
        self.pca_variance = pca_variance
        self.pipeline = None  # Will store the full ColumnTransformer
        self.pca = None       # Optional PCA step
        self.meta_features_ = {}  # Stores dataset-level meta-features

    def fit(self, X, y=None):
        """
        Fits the preprocessing pipeline to the input features.

        Args:
            X (pd.DataFrame): Feature matrix
            y (optional): Not used here

        Returns:
            self
        """
        X = X.copy()

        # Identify numerical and categorical columns
        self.num_cols = X.select_dtypes(include=np.number).columns.tolist()
        self.cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

        # Define pipelines for each type of feature
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])

        # Combine both into a ColumnTransformer
        self.pipeline = ColumnTransformer([
            ('num', num_pipeline, self.num_cols),
            ('cat', cat_pipeline, self.cat_cols)
        ])

        # Fit the transformer and optionally apply PCA
        Xt = self.pipeline.fit_transform(X)

        if self.use_pca:
            self.pca = PCA(n_components=self.pca_variance)
            self.pca.fit(Xt)

        # Extract and store useful dataset-level statistics
        self.meta_features_ = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_numeric": len(self.num_cols),
            "n_categorical": len(self.cat_cols),
            "missing_values": X.isnull().sum().sum(),
        }

        return self

    def transform(self, X):
        """
        Transforms new data using the fitted pipeline.

        Args:
            X (pd.DataFrame): Input feature matrix

        Returns:
            np.ndarray: Transformed feature matrix
        """
        X = X.copy()
        Xt = self.pipeline.transform(X)

        if self.use_pca and self.pca is not None:
            Xt = self.pca.transform(Xt)

        return Xt

    def fit_transform(self, X, y=None):
        """
        Convenience method: fits and transforms in one step.

        Args:
            X (pd.DataFrame): Input data

        Returns:
            np.ndarray: Transformed feature matrix
        """
        return self.fit(X, y).transform(X)
