import logging
import numpy as np
import optuna

from sklearn.linear_model import Ridge
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, KFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from src.automl.preprocessing import Preprocessor

logger = logging.getLogger(__name__)


class AutoML:
    def __init__(self, preprocessing: Preprocessor = None, seed: int = 42):
        """
        AutoML class that orchestrates preprocessing, model training,
        hyperparameter optimization, and ensembling.

        Args:
            preprocessing (Preprocessor): Optional preprocessing pipeline.
            seed (int): Random seed for reproducibility.
        """
        self.seed = seed
        self.preprocessing = preprocessing or Preprocessor()
        self.models = {}
        self.cat_feature_indices = []
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def _evaluate(self, model, X, y):
        """
        Evaluate a model using 5-fold cross-validation and return mean R².

        Args:
            model: Scikit-learn-compatible regressor.
            X: Preprocessed feature matrix.
            y: Target values.

        Returns:
            float: Mean R² score.
        """
        cv = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)
        return np.mean(scores)

    def _build_models(self, X_transformed, y):
        """
        Constructs and returns a dictionary of models including Ridge,
        LightGBM with HPO, and CatBoost (with native categorical support).

        Args:
            X_transformed: Preprocessed feature matrix (excluding CatBoost).
            y: Target values.

        Returns:
            dict: Dictionary of model names and model instances.
        """
        meta = self.preprocessing.meta_features_
        self.logger.info(f"Meta-features used for HPO: {meta}")

        # === Ridge Regression (baseline model) ===
        ridge = Ridge(alpha=1.0)

        # === LightGBM with Optuna Hyperparameter Optimization ===
        n_samples = meta.get("n_samples", X_transformed.shape[0])
        n_features = meta.get("n_features", X_transformed.shape[1])
        cardinality = meta.get("mean_cardinality", 10)

        max_estimators = min(500, int(n_samples / 2))
        max_depth_upper = min(16, int(cardinality + n_features / 2))

        def objective_lgb(trial):
            return LGBMRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, max_estimators),
                max_depth=trial.suggest_int("max_depth", 3, max_depth_upper),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                random_state=self.seed,
            )

        study_lgb = optuna.create_study(direction="maximize")
        study_lgb.enqueue_trial({
            "n_estimators": min(100, max_estimators),
            "max_depth": min(6, max_depth_upper),
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        })
        study_lgb.optimize(lambda trial: self._evaluate(objective_lgb(trial), X_transformed, y), n_trials=10)
        lgb_model = objective_lgb(study_lgb.best_trial)

        self.logger.info(f"Best LightGBM trial: {study_lgb.best_trial.params}")

        # === CatBoost with native support for categorical features ===
        X_raw = self.preprocessing.X_original  # Access original DataFrame
        self.cat_feature_indices = self.preprocessing.get_catboost_feature_indices(X_raw)
        cat_model = CatBoostRegressor(verbose=0, random_seed=self.seed)
        cat_model.fit(X_raw, y, cat_features=self.cat_feature_indices)

        return {
            "ridge": ridge,
            "lightgbm": lgb_model,
            "catboost": cat_model
        }

    def fit(self, X, y):
        """
        Fits the preprocessing, builds all models, and trains the final ensemble.

        Args:
            X: Raw training features (DataFrame).
            y: Training targets.
        """
        logger.info("Fitting preprocessing...")
        X_transformed = self.preprocessing.fit_transform(X)
        logger.info("Building and fitting models...")
        self.models = self._build_models(X_transformed, y)

        for name, model in self.models.items():
            if name == "catboost":
                model.fit(X, y, cat_features=self.cat_feature_indices)
                preds = model.predict(X)
            else:
                model.fit(X_transformed, y)
                preds = model.predict(X_transformed)

            score = r2_score(y, preds)
            logger.info(f"Model {name} R² on train: {score:.4f}")

        # === Ensemble using VotingRegressor (excluding CatBoost if incompatible) ===
        self.ensemble = VotingRegressor(
            estimators=[(name, model) for name, model in self.models.items()]
        )
        self.ensemble.fit(X_transformed, y)
        logger.info(f"Final ensemble contains: {list(self.models.keys())}")

    def predict(self, X):
        """
        Predicts on new data using the fitted ensemble.

        Args:
            X: New test data (DataFrame).

        Returns:
            np.ndarray: Predicted target values.
        """
        X_transformed = self.preprocessing.transform(X)
        return self.ensemble.predict(X_transformed)

    def get_metadata(self):
        """
        Returns summary information about the current pipeline run.

        Returns:
            dict: Dictionary containing model names and meta-features.
        """
        meta_features = {
            k: int(v) if isinstance(v, (np.integer, np.int32, np.int64)) else float(v)
            for k, v in getattr(self.preprocessing, "meta_features_", {}).items()
        }

        return {
            "models": list(self.models.keys()),
            "meta_features": meta_features,
        }
