from sklearn.linear_model import Ridge
from sklearn.ensemble import VotingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import optuna
import logging
from src.automl.preprocessing import Preprocessor

logger = logging.getLogger(__name__)


class AutoML:
    def __init__(self, preprocessing: Preprocessor = None, seed: int = 42):
        self.seed = seed
        self.preprocessing = preprocessing or Preprocessor()
        self.models = {}
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def _build_models(self, X, y):
        """Builds multiple models with tuned hyperparameters."""

        ridge = Ridge(alpha=1.0)

        def objective_lgb(trial):
            return LGBMRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 3, 12),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                random_state=42
            )

        study_lgb = optuna.create_study(direction="maximize")
        study_lgb.enqueue_trial({"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8})
        study_lgb.optimize(lambda trial: self._evaluate(objective_lgb(trial), X, y), n_trials=10)

        lgb_model = objective_lgb(study_lgb.best_trial)

        cat_model = CatBoostRegressor(verbose=0, random_seed=42)

        return {"ridge": ridge, "lightgbm": lgb_model, "catboost": cat_model}

    def _evaluate(self, model, X, y):
        """Simple evaluation using holdout split for tuning."""
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        return model.score(X_val, y_val)

    def fit(self, X, y):
        logger.info("Fitting preprocessing...")
        X_transformed = self.preprocessing.fit_transform(X)
        logger.info("Fitting models...")
        self.models = self._build_models(X_transformed, y)

        for name, model in self.models.items():
            logger.info(f"Training model: {name}")
            model.fit(X_transformed, y)

        self.ensemble = VotingRegressor(estimators=[(k, m) for k, m in self.models.items()])
        self.ensemble.fit(X_transformed, y)

    def predict(self, X):
        X_transformed = self.preprocessing.transform(X)
        return self.ensemble.predict(X_transformed)
