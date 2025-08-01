import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Dict, Any, Optional

logger = logging.getLogger("automl")


class Preprocessor:

    def __init__(self, seed: int):
        self.seed = seed
        self.numerical_features: list[str] = []
        self.categorical_features: list[str] = []
        self.preprocessor: Optional[ColumnTransformer] = None
        self.missing_value_replacements: Dict[str, Any] = {}

    def _identify_feature_types(self, X: pd.DataFrame) -> None:
        self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

        logger.info(f"Identified {len(self.numerical_features)} numerical and "
                    f"{len(self.categorical_features)} categorical features")

    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        X_filled = X.copy()

        for col in self.numerical_features:
            if X_filled[col].isnull().any():
                filler = X_filled[col].median()
                self.missing_value_replacements[col] = filler
                X_filled[col].fillna(filler, inplace=True)

        for col in self.categorical_features:
            if X_filled[col].isnull().any():
                mode_values = X_filled[col].mode()
                filler = mode_values.iloc[0] if len(mode_values) > 0 else 'missing'
                self.missing_value_replacements[col] = filler
                X_filled[col].fillna(filler, inplace=True)

        return X_filled

    def _create_preprocessor(self) -> ColumnTransformer:
        transforms = []

        if self.numerical_features:
            numerical_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            transforms.append(('num', numerical_transformer, self.numerical_features))

        if self.categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transforms.append(('cat', categorical_transformer, self.categorical_features))

        return ColumnTransformer(transformers=transforms, remainder='passthrough')

    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Fitting preprocessor...")
        self._identify_feature_types(X)
        X_corrected = self._handle_missing_values(X)

        self.preprocessor = self._create_preprocessor()
        X_transformed = self.preprocessor.fit_transform(X_corrected)

        if hasattr(self.preprocessor, 'get_feature_names_out'):
            feature_names = self.preprocessor.get_feature_names_out()
        else:
            # Fallback for older sklearn versions
            feature_names = [f'feature_{i}' for i in range(X_transformed.shape[1])]

        X_df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

        logger.info(f"Preprocessing completed. Shape: {X_df.shape}")
        return X_df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted")

        X_filled = X.copy()
        for col, filler in self.missing_value_replacements.items():
            if col in X_filled.columns and X_filled[col].isnull().any():
                X_filled[col].fillna(filler, inplace=True)

        X_transformed = self.preprocessor.transform(X_filled)

        if hasattr(self.preprocessor, 'get_feature_names_out'):
            feature_names = self.preprocessor.get_feature_names_out()
        else:
            feature_names = [f'feature_{i}' for i in range(X_transformed.shape[1])]

        return pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
