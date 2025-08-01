import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, Any, Optional

logger = logging.getLogger("automl")


class FrequencyTargetEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.encodings: Dict[str, Dict] = {}
        self.global_mean: float = 0.0
        self.categorical_columns: list[str] = []

    def fit(self, X: pd.DataFrame, Y: Optional[pd.Series] = None):
        self.categorical_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()

        if Y is None or len(self.categorical_columns) == 0:
            # Frequency encoding only
            for col in self.categorical_columns:
                self.encodings[col] = X[col].value_counts().to_dict()
        else:
            # Target encoding with smoothing
            self.global_mean = Y.mean()
            for col in self.categorical_columns:
                grouped = pd.DataFrame({'target': Y, 'cat': X[col]}).groupby('cat')
                counts = grouped.size()
                means = grouped['target'].mean()

                # Apply smoothing
                smoothed_means = (counts * means + self.smoothing * self.global_mean) / (counts + self.smoothing)
                self.encodings[col] = smoothed_means.to_dict()

        return self

    def transform(self, X: pd.DataFrame):
        X_encoded = X.copy()
        for col in self.categorical_columns:
            if col in X_encoded.columns:
                encoding = self.encodings.get(col, {})
                default_value = self.global_mean if self.global_mean else 0
                X_encoded[col] = X_encoded[col].map(encoding).fillna(default_value)
        return X_encoded


class FeatureEngineer:
    def __init__(self, seed: int):
        self.seed = seed
        self.meta_features: Dict[str, Any] = {}
        self.freq_target_encoder: Optional[FrequencyTargetEncoder] = None
        self.poly_features: Optional[PolynomialFeatures] = None
        self.use_polynomial: bool = False
        self.original_columns: list[str] = []

    def _extract_meta_features(self, X: pd.DataFrame, Y: Optional[pd.Series] = None) -> Dict[str, Any]:
        meta_features = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'missing_ratio': X.isnull().sum().sum() / (X.shape[0] * X.shape[1]),
        }

        numerical_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns

        meta_features.update({
            'numerical_ratio': len(numerical_cols) / X.shape[1] if X.shape[1] > 0 else 0,
            'categorical_ratio': len(categorical_cols) / X.shape[1] if X.shape[1] > 0 else 0,
        })

        if len(numerical_cols) > 0:
            meta_features.update({
                'mean_skewness': X[numerical_cols].skew().mean(),
                'mean_kurtosis': X[numerical_cols].kurtosis().mean(),
                'correlation_mean': np.abs(X[numerical_cols].corr()).mean().mean(),
            })

        if Y is not None:
            meta_features.update({
                'target_skewness': Y.skew(),
                'target_kurtosis': Y.kurtosis(),
                'target_std': Y.std(),
            })

        return meta_features

    def _apply_frequency_target_encoding(self, X: pd.DataFrame, Y: Optional[pd.Series] = None) -> pd.DataFrame:
        if self.freq_target_encoder is None:
            self.freq_target_encoder = FrequencyTargetEncoder(smoothing=1.0)
            self.freq_target_encoder.fit(X, Y)

        return self.freq_target_encoder.transform(X)

    def _apply_polynomial_features(self, X: pd.DataFrame) -> pd.DataFrame:
        # Only apply if reasonable number of features to avoid memory explosion
        numerical_cols = X.select_dtypes(include=[np.number])

        if numerical_cols.shape[1] <= 20 and numerical_cols.shape[1] > 0:
            self.use_polynomial = True
            if self.poly_features is None:
                self.poly_features = PolynomialFeatures(
                    degree=2,
                    interaction_only=True,
                    include_bias=False
                )
                X_poly = self.poly_features.fit_transform(numerical_cols)
            else:
                X_poly = self.poly_features.transform(numerical_cols)

            # Create DataFrame with polynomial features
            poly_feature_names = self.poly_features.get_feature_names_out(numerical_cols.columns)
            X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=X.index)

            # Combine with non-numerical columns
            non_numerical_cols = X.select_dtypes(exclude=[np.number])
            if not non_numerical_cols.empty:
                X_combined = pd.concat([non_numerical_cols, X_poly_df], axis=1)
            else:
                X_combined = X_poly_df

            return X_combined
        else:
            self.use_polynomial = False
            return X

    def fit(self, X: pd.DataFrame, Y: pd.Series) -> pd.DataFrame:
        logger.info("Starting feature engineering...")

        self.original_columns = X.columns.tolist()

        # Extract meta features
        self.meta_features = self._extract_meta_features(X, Y)
        logger.info(f"Meta features: {self.meta_features}")

        # Apply frequency/target encoding
        X_encoded = self._apply_frequency_target_encoding(X, Y)

        # Apply polynomial features
        X_poly = self._apply_polynomial_features(X_encoded)

        logger.info(f"Feature engineering completed. Shape: {X_poly.shape}")
        return X_poly

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_encoded = self._apply_frequency_target_encoding(X)

        # Apply polynomial features if used during training
        if self.use_polynomial:
            X_poly = self._apply_polynomial_features(X_encoded)
        else:
            X_poly = X_encoded

        return X_poly

    def get_meta_features(self) -> Dict[str, Any]:
        return self.meta_features
