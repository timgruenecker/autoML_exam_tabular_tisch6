import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Union, Optional
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import time

logger = logging.getLogger("automl")


class ModelConfig:
    def __init__(
            self, model_type: str, params: Dict[str, Any], score: float,
            model: Union[Ridge, lgb.LGBMRegressor, cb.CatBoostRegressor, xgb.XGBRegressor],
            cv_scores: Optional[List[float]] = None):
        self.model_type = model_type
        self.params = params
        self.score = score
        self.model = model
        self.cv_scores = cv_scores if cv_scores is not None else []
        self.std_score = np.std(self.cv_scores) if self.cv_scores else 0.0


class MultiObjectiveResult:
    def __init__(self, config: ModelConfig, training_time: float, complexity: float, pareto_rank: int = 0):
        self.config = config
        self.training_time = training_time
        self.complexity = complexity
        self.pareto_rank = pareto_rank


class HyperparameterOptimizer:
    def __init__(self, seed: int, time_budget: int, n_trials: int, cv_folds: int = 5):
        self.seed = seed
        self.time_budget = time_budget
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.meta_features: Dict[str, Any] = {}
        self.start_time = None

        self.min_budget = 0.1
        self.max_budget = 1.0
        self.eta = 3

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

        elif model_type == 'xgboost':
            return {
                "n_estimators": trial.suggest_int('n_estimators', 50, min(2000, max(100, n_samples // 2))),
                "max_depth": trial.suggest_int('max_depth', 3, min(15, int(np.log2(max(n_features, 2))) + 5)),
                "learning_rate": trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                "subsample": trial.suggest_float('subsample', 0.6, 1.0),
                "colsample_bytree": trial.suggest_float('colsample_bytree', 0.6, 1.0),
                "reg_alpha": trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                "min_child_weight": trial.suggest_int('min_child_weight', 1, min(20, max(5, n_samples // 100))),
                "gamma": trial.suggest_float('gamma', 1e-8, 1.0, log=True)
            }

        elif model_type == 'catboost':
            return {
                'iterations': trial.suggest_int('iterations', 50, min(2000, max(100, n_samples // 2))),
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

        elif model_type == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, min(500, max(100, n_samples // 2))),
                'max_depth': trial.suggest_int('max_depth', 3, min(20, int(np.log2(max(n_features, 2))) + 8)),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, min(20, max(5, n_samples // 100))),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, min(10, max(3, n_samples // 200))),
                'max_features': trial.suggest_float('max_features', 0.3, 1.0),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            }

        elif model_type == 'extra_trees':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, min(500, max(100, n_samples // 10))),
                'max_depth': trial.suggest_int('max_depth', 3, min(20, int(np.log2(max(n_features, 2))) + 8)),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, min(20, max(5, n_samples // 100))),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, min(10, max(3, n_samples // 200))),
                'max_features': trial.suggest_float('max_features', 0.3, 1.0),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
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

        elif model_type == 'xgboost':
            try:
                model = xgb.XGBRegressor(
                    tree_method="gpu_hist",
                    predictor="gpu_predictor",
                    random_state=self.seed,
                    verbosity=0,
                    **params
                )
            except:
                model = xgb.XGBRegressor(
                    tree_method="hist",
                    predictor="auto",
                    random_state=self.seed,
                    verbosity=0,
                    **params
                )
            return model

        elif model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                random_state=self.seed,
                n_jobs=1,
                **params
            )

        elif model_type == 'extra_trees':
            from sklearn.ensemble import ExtraTreesRegressor
            return ExtraTreesRegressor(
                random_state=self.seed,
                n_jobs=1,
                **params
            )

        elif model_type == 'elastic_net':
            from sklearn.linear_model import ElasticNet
            return ElasticNet(
                random_state=self.seed,
                **params
            )

        raise ValueError(f"Unknown model type: {model_type}")

    def _get_budget_subset(self, X: pd.DataFrame, y: pd.Series, budget: float) -> Tuple[pd.DataFrame, pd.Series]:
        if budget >= 1.0:
            return X, y

        n_samples = int(len(X) * budget)
        n_samples = max(n_samples, 50)

        indices = np.random.RandomState(self.seed).choice(len(X), size=n_samples, replace=False)
        return X.iloc[indices], y.iloc[indices]

    def _objective_with_fidelity(
            self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, model_type: str, budget: float) -> float:
        try:
            if self.start_time and (time.time() - self.start_time) > self.time_budget:
                logger.info("Time budget exceeded, stopping optimization")
                trial.study.stop()
                return -np.inf

            X_subset, y_subset = self._get_budget_subset(X, y, budget)
            params = self._get_search_space(trial, model_type)

            if model_type in ['lightgbm', 'catboost', 'xgboost']:
                if 'n_estimators' not in params:
                    params['n_estimators'] = int(params['n_estimators']*budget)
                    params['n_estimators'] = max(params['n_estimators'], 10)
                if 'iterations' in params:
                    params['iterations'] = int(params['iterations']*budget)
                    params['iterations'] = max(params['iterations'], 10)

            model = self._create_model(model_type, params)

            cv_folds = max(3, int(self.cv_folds * budget))
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.seed)

            scores = cross_val_score(model, X_subset, y_subset, cv=cv, scoring='r2', n_jobs=1)
            score_penalty = 1.0 - (1.0 - budget) * 0.1

            return np.mean(scores) * score_penalty

        except Exception as e:
            logger.warning(f"Trial failed for {model_type} with budget {budget}: {e}")
            return -np.inf

    def _successive_halving_optimize(
            self, X: pd.DataFrame, y: pd.Series, model_type: str, trials_per_model: int) -> ModelConfig:
        logger.info(f"Starting successive halving optimization for {model_type}")

        budgets = [0.1, 0.3, 0.6, 1.0]
        configs_per_budget = [trials_per_model, trials_per_model // 2, trials_per_model // 4, trials_per_model // 8]
        configs_per_budget = [max(c, 5) for c in configs_per_budget]

        best_configs = []

        for budget, n_configs in zip(budgets, configs_per_budget):
            logger.info(f"Evaluating {n_configs} configurations at budget {budget}")

            study = optuna.create_study(
                study_name=f"{model_type}_budget_{budget}",
                direction='maximize',
                sampler=TPESampler(seed=self.seed),
                pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=5)
            )
            study.optimize(
                lambda trial: self._objective_with_fidelity(trial, X, y, model_type, budget),
                n_trials=n_configs,
                show_progress_bar=False
            )
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE and trial.value > -np.inf:
                    best_configs.append((trial.value, trial.params, budget))

            if self.start_time and (time.time() - self.start_time) > self.time_budget * 0.8:
                logger.info("Approaching time budget limit, stopping successive halving")
                break

        if not best_configs:
            logger.warning(f"No successful configurations found for {model_type}")
            default_params = self._get_default_params(model_type)
            model = self._create_model(model_type, default_params)
            return ModelConfig(model_type, default_params, -1.0, model)

        # # TODO: Buggy line, reorder tuple extraction probably would fix it
        # best_configs = sorted(best_configs, key=lambda x: x[0], reverse=True)
        best_score, best_params, best_budget = max(best_configs, key=lambda x: x[0])

        final_model = self._create_model(model_type, best_params)

        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        final_scores = cross_val_score(final_model, X, y, cv=cv, scoring='r2', n_jobs=1)
        final_score = np.mean(final_scores)

        final_model.fit(X, y)

        logger.info(f"{model_type} final score: {final_score:.4f} (std: {np.std(final_scores):.4f}) ")

        return ModelConfig(model_type, best_params, final_score, final_model, cv_scores=final_scores.tolist())

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

    def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        defaults = {
            "lightgbm": {'n_estimators': 100, 'learning_rate': 0.1},
            "xgboost": {'n_estimators': 100, 'learning_rate': 0.1},
            "catboost": {'iterations': 100, 'learning_rate': 0.1},
            "ridge": {'alpha': 1.0},
            "random_forest": {'n_estimators': 100},
            "extra_trees": {'n_estimators': 100},
            "elastic_net": {'alpha': 1.0, 'l1_ratio': 0.5}
        }
        return defaults.get(model_type, {})

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

        self.start_time = time.time()
        self.meta_features = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
        }

        logger.info(f"Starting DEHB-inspired HPO for {len(model_types)} models...")
        logger.info(f"Time budget: {self.time_budget} seconds")
        logger.info(f"Meta features: {self.meta_features}")

        optimized_models = []
        time_per_model = self.time_budget // len(model_types)
        trials_per_model = max(20, self.n_trials // len(model_types))

        for i, model_type in enumerate(model_types):
            model_start_time = time.time()
            try:
                elapsed_time = time.time() - self.start_time
                if elapsed_time > self.time_budget * 0.9:
                    logger.info("Approaching time budget limit, stopping optimization")
                    break

                model_config = self._successive_halving_optimize(X, y, model_type, trials_per_model)
                optimized_models.append(model_config)

                model_time = time.time() - model_start_time
                logger.info(f"Optimized {model_type} in {model_time:.2f}")

            except Exception as e:
                logger.error(f"Failed to optimize {model_type}: {e}")

        optimized_models.sort(key=lambda x: x.score, reverse=True)

        if optimized_models:
            logger.info(
                f"HPO completed. Best model: {optimized_models[0].model_type} with score {optimized_models[0].score:.4f} (std: {optimized_models[0].std_score:.4f})")
        else:
            logger.warning("No models were successfully optimized.")

        total_time = time.time() - self.start_time
        logger.info(f"Total optimization time: {total_time:.2f} seconds")

        return optimized_models
