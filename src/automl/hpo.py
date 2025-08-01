import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Union
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import lightgbm as lgb
import catboost as cb

logger = logging.getLogger("automl")


class ModelConfig:
    def __init__(
            self, model_type: str, params: Dict[str, Any],
            score: float, model: Union[Ridge, lgb.LGBMRegressor, cb.CatBoostRegressor]):
        self.model_type = model_type
        self.params = params
        self.score = score
        self.model = model


class HyperparameterOptimizer:
    def __init__(self, seed: int, time_budget: int, n_trials: int):
        self.seed = seed
        self.time_budget = time_budget
        self.n_trials = n_trials
        self.meta_features: Dict[str, Any] = {}

    def _get_search_space(self, trial: optuna.Trial, model_type: str) -> Dict[str, Any]:
        n_samples = self.meta_features.get('n_samples', 1000)
        n_features = self.meta_features.get('n_features', 10)

        if model_type == 'lightgbm':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, min(1000, max(100, n_samples // 2))),
                'max_depth': trial.suggest_int('max_depth', 3, min(15, int(np.log2(max(n_features, 2))) + 5)),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 10, min(300, 2**min(10, int(np.log2(max(n_features, 2))) + 3))),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, min(100, max(20, n_samples // 100))),
            }

        elif model_type == 'catboost':
            return {
                'iterations': trial.suggest_int('iterations', 50, min(1000, max(100, n_samples // 2))),
                'depth': trial.suggest_int('depth', 3, min(10, int(np.log2(max(n_features, 2))) + 3)),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, min(50, max(10, n_samples // 100))),
            }

        elif model_type == 'ridge':
            # Adjust alpha range based on dataset size
            alpha_min = 1e-8 if n_samples > n_features else 1e-4
            alpha_max = 1000.0 if n_samples > n_features else 10000.0

            return {
                'alpha': trial.suggest_float('alpha', alpha_min, alpha_max, log=True),
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            }

        return {}

    def _create_model(self, model_type: str, params: Dict[str, Any]):
        """Create model instance with given parameters"""

        if model_type == 'lightgbm':
            return lgb.LGBMRegressor(
                random_state=self.seed,
                verbose=-1,
                force_col_wise=True,
                **params
            )

        elif model_type == 'catboost':
            return cb.CatBoostRegressor(
                random_state=self.seed,
                verbose=False,
                **params
            )

        elif model_type == 'ridge':
            return Ridge(
                random_state=self.seed,
                **params
            )

        raise ValueError(f"Unknown model type: {model_type}")

    def _objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, model_type: str) -> float:
        """Objective function for hyperparameter optimization"""

        try:
            # Get hyperparameters
            params = self._get_search_space(trial, model_type)

            # Create model
            model = self._create_model(model_type, params)

            # Cross-validation with early stopping for tree models
            if model_type in ['lightgbm', 'catboost']:
                # Use smaller CV for speed during HPO
                scores = cross_val_score(model, X, y.values, cv=3, scoring='r2', n_jobs=1)
            else:
                scores = cross_val_score(model, X, y.values, cv=5, scoring='r2', n_jobs=1)

            return np.mean(scores)

        except Exception as e:
            logger.warning(f"Trial failed for {model_type}: {e}")
            return -np.inf

    def _optimize_single_model(self, X: pd.DataFrame, y: pd.Series, model_type: str) -> ModelConfig:
        """Optimize hyperparameters for a single model type"""

        logger.info(f"Optimizing {model_type}...")

        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.seed),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )

        # Calculate trials per model
        trials_per_model = max(10, self.n_trials // 4)  # At least 10 trials per model
        time_per_model = self.time_budget // 4

        # Optimize
        study.optimize(
            lambda trial: self._objective(trial, X, y, model_type),
            n_trials=trials_per_model,
            timeout=time_per_model,
            show_progress_bar=False
        )

        best_params = study.best_params
        best_score = study.best_value

        # Train final model with best parameters
        best_model = self._create_model(model_type, best_params)
        best_model.fit(X, y)

        logger.info(f"{model_type} best score: {best_score:.4f}")

        return ModelConfig(
            model_type=model_type,
            params=best_params,
            score=best_score,
            model=best_model
        )

    def optimize_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_types: List[str]
    ) -> List[ModelConfig]:
        """Optimize hyperparameters for all selected models"""

        # Store meta features for search space adaptation
        self.meta_features = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
        }

        logger.info(f"Starting HPO for {len(model_types)} models with meta features: {self.meta_features}")

        optimized_models = []

        for model_type in model_types:
            try:
                model_config = self._optimize_single_model(X, y, model_type)
                optimized_models.append(model_config)
            except Exception as e:
                logger.error(f"Failed to optimize {model_type}: {e}")

        # Sort by performance
        optimized_models.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"HPO completed. Best model: {optimized_models[0].model_type} "
                    f"with score: {optimized_models[0].score:.4f}")

        return optimized_models
