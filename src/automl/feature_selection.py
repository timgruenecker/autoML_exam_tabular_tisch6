import pandas as pd
import numpy as np
import logging
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from typing import Optional

logger = logging.getLogger("automl")

# TODO: Many pipeline hyperparams here -> set dynamically?


class FeatureSelector:
    def __init__(self, seed: int):
        self.seed = seed
        self.feature_selector: Optional[Pipeline] = None
        self.selected_feature_names: list[str] = []

    def _create_selector_pipeline(self, X: pd.DataFrame, Y: pd.Series) -> Pipeline:
        steps = []

        variance_selector = VarianceThreshold(threshold=0.01)
        steps.append(('variance', variance_selector))

        lgb_selector = lgb.LGBMRegressor(
            random_state=self.seed,
            verbose=-1,
            n_estimators=50,
            force_col_wise=True
        )
        importance_selector = SelectFromModel(lgb_selector, threshold='median')
        steps.append(('importance', importance_selector))

        pipeline = Pipeline(steps)
        pipeline.fit(X, Y)
        X_temp = pipeline.transform(X)

        if X_temp.shape[1] > 100:
            logger.info(f"Adding PCA due to high dimensionality: {X_temp.shape[1]} features")
            pca = PCA(n_components=0.95, random_state=self.seed)  # Keep 95% variance
            steps.append(('pca', pca))
            pipeline = Pipeline(steps)

        return pipeline

    def select_features(self, X: pd.DataFrame, Y: pd.Series) -> pd.DataFrame:
        logger.info(f"Starting feature selection on {X.shape[1]} features...")

        X_values = X.values

        self.feature_selector = self._create_selector_pipeline(X, Y)
        X_selected = self.feature_selector.fit_transform(X, Y)

        n_selected = X_selected.shape[1]
        feature_names = [f'selected_feature_{i}' for i in range(n_selected)]
        self.selected_feature_names = feature_names

        X_selected_df = pd.DataFrame(
            X_selected,
            columns=feature_names,
            index=X.index
        )

        logger.info(f"Feature selection completed. Selected {n_selected} features from {X.shape[1]}")

        return X_selected_df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_selector is None:
            raise ValueError("Feature selector not fitted")

        X_values = X.values

        X_selected = self.feature_selector.transform(X)

        X_selected_df = pd.DataFrame(
            X_selected,
            columns=self.selected_feature_names,
            index=X.index
        )

        return X_selected_df

    def get_selected_feature_count(self) -> int:
        return len(self.selected_feature_names)
