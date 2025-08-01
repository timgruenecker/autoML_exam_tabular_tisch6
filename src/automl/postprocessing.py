import pandas as pd
import numpy as np
import logging
from typing import List
from .hpo import ModelConfig
from .preprocessing import Preprocessor
from .feature_engineering import FeatureEngineer
from .feature_selection import FeatureSelector

logger = logging.getLogger("automl")


class EnsembleModel:

    def __init__(
        self,
        models: List[ModelConfig],
        preprocessor: Preprocessor,
        feature_engineer: FeatureEngineer,
        feature_selector: FeatureSelector
    ):
        self.models = models
        self.preprocessor = preprocessor
        self.feature_engineer = feature_engineer
        self.feature_selector = feature_selector
        self.weights = self._calculate_weights()

        logger.info(f"Created ensemble with {len(models)} models")
        for i, model in enumerate(models):
            logger.info(f"  Model {i+1}: {model.model_type} (score: {model.score:.4f}, weight: {self.weights[i]:.3f})")

    def _calculate_weights(self) -> np.ndarray:
        """Calculate ensemble weights based on model scores"""
        if not self.models:
            return np.array([])

        scores = np.array([model.score for model in self.models])

        # Handle negative scores by shifting
        if np.any(scores < 0):
            scores = scores - np.min(scores) + 1e-8

        # Simple performance-based weighting
        weights = scores / np.sum(scores)

        return weights

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions"""
        if not self.models:
            raise ValueError("No models in ensemble")

        # # Apply the same preprocessing pipeline as training
        # X_preprocessed = self.preprocessor.transform(X)
        # X_engineered = self.feature_engineer.transform(X_preprocessed)
        # X_selected = self.feature_selector.transform(X_engineered)

        # Generate predictions from all models
        predictions = []
        for model_config in self.models:
            pred = model_config.model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)

        # Weighted average ensemble
        if len(self.weights) > 0:
            final_predictions = np.average(predictions, axis=0, weights=self.weights)
        else:
            # Fallback to simple average
            final_predictions = np.mean(predictions, axis=0)

        return final_predictions
