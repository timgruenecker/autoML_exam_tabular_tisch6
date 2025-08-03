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
        self.use_ensemble = len(models) > 1

        logger.info(f"Created ensemble with {len(models)} models")
        for i, model in enumerate(models):
            logger.info(f"  Model {i+1}: {model.model_type} (score: {model.score:.4f}, weight: {self.weights[i]:.3f})")

    def _calculate_weights(self) -> np.ndarray:
        if not self.models:
            return np.array([])

        scores = np.array([model.score for model in self.models])

        if len(scores) == 1:
            return np.array([1.0])

        logger.info(f"Models scores for weighting: {scores}")

        sufficient_models = scores > -0.5
        if not np.any(sufficient_models):
            best_idx = np.argmax(scores)
            sufficient_models = np.zeros_like(scores, dtype=bool)
            sufficient_models[best_idx] = True
            logger.warning("All models are dogshit, using only the best model for ensemble")

        good_scores = scores[sufficient_models]
        self.models = [model for i, model in enumerate(self.models) if sufficient_models[i]]

        if len(good_scores) == 1:
            logger.info("Only one good model, using it as the ensemble")
            self.use_ensemble = False
            return np.array([1.0])

        min_score = np.min(good_scores)
        shifted_scores = good_scores - min_score + 1e-6

        temperature = 2.0
        exp_scores = np.exp(shifted_scores / temperature)
        weights = exp_scores / np.sum(exp_scores)

        if np.std(good_scores) < 0.01:
            logger.info("All models have very similar scores, using rank-based weights")
            ranks = np.argsort(np.argsort(-good_scores)) + 1
            weights = ranks / np.sum(ranks)

        weights = np.abs(weights)
        weights = weights / np.sum(weights)

        for i, (model, weight) in enumerate(zip(self.models, weights)):
            logger.info(f"Final weight for {model.model_type}: (score: {model.score:.4f}, weight: {weight:.3f})")

        return weights

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions"""
        if not self.models:
            raise ValueError("No models in ensemble")

        if not self.use_ensemble:
            return self.models[0].model.predict(X)

        predictions = []
        valid_predictions = []
        valid_weights = []

        for i, model_config in enumerate(self.models):
            try:
                pred = model_config.model.predict(X)
                if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                    logger.warning(f"Model {model_config.model_type} produced invalid predictions")
                    continue

                predictions.append(pred)
                valid_predictions.append(pred)
                valid_weights.append(self.weights[i])

            except Exception as e:
                logger.error(f"Model {model_config.model_type} failed to predict: {e}")

        if not valid_predictions:
            raise ValueError("No valid predictions from models")

        if len(valid_predictions) == 1:
            logger.info("Only one valid model prediction, returning it directly")
            return valid_predictions[0]

        predictions = np.array(valid_predictions)
        valid_weights = np.array(valid_weights)
        valid_weights = valid_weights / np.sum(valid_weights)

        final_predictions = np.average(predictions, axis=0, weights=valid_weights)

        logger.info(f"Ensemble predictions generated with {len(valid_predictions)} valid models")

        return final_predictions
