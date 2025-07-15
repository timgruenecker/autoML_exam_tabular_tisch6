import logging
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self):
        self._models = []

        # Register basic models
        self.register("Ridge", Ridge())
        self.register("RandomForest", RandomForestRegressor())
        self.register("XGBoost", XGBRegressor(objective="reg:squarederror", verbosity=0))

    def register(self, name, model):
        self._models.append((name, model))
        logger.info(f"Registered model: {name}")

    def get_models(self):
        return self._models
