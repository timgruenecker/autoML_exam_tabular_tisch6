import logging
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self):
        """
        Initializes the model registry and registers common models.
        """
        self._models = []

        # Register some commonly used models
        self.register("Ridge", Ridge())
        self.register("RandomForest", RandomForestRegressor())
        self.register("XGBoost", XGBRegressor(objective="reg:squarederror", verbosity=0))

    def register(self, name, model):
        """
        Registers a new model to the registry.

        Args:
            name (str): Name of the model
            model (sklearn estimator): Model object
        """
        self._models.append((name, model))
        logger.info(f"Registered model: {name}")

    def get_models(self):
        """
        Returns all registered models as (name, model) pairs.

        Returns:
            list of tuples: (model_name, model_instance)
        """
        return self._models
