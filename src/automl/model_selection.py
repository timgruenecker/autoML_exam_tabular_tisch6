import logging
from typing import Dict, Any, List

logger = logging.getLogger("automl")


class ModelSelector:
    def __init__(self, seed: int):
        self.seed = seed

    def select_models(self, meta_features: Dict[str, Any]) -> List[str]:
        selected_models = []
        selected_models.append('lightgbm')
        selected_models.append('ridge')

        categorical_ratio = meta_features.get('categorical_ratio', 0)
        if categorical_ratio > 0.1:
            selected_models.append('catboost')
            logger.info(f"Adding CatBoost due to categorical ratio: {categorical_ratio:.3f}")

        n_samples = meta_features.get('n_samples', 0)
        n_features = meta_features.get('n_features', 0)

        if n_samples < 1000:
            logger.info(f"Small dataset detected(n_samples={n_samples}), using simpler model set")

        if n_features > n_samples:
            logger.info(f"High-dimensional dataset detected (n_features={n_features}, n_samples={n_samples}")

        logger.info(f"Selected models: {selected_models}")
        return selected_models
