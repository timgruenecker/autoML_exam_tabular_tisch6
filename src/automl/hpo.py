import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Union, Optional
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import time

from dehb import DEHB
from ConfigSpace import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    CategoricalHyperparameter
)
from ConfigSpace.conditions import EqualsCondition
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

logger = logging.getLogger("automl")


class ModelConfig:
    def __init__(
            self, model_type: str, params: Dict[str, Any], score: float,
            model: Union[Ridge, lgb.LGBMRegressor, cb.CatBoostRegressor, xgb.XGBRegressor],
            cv_scores: Optional[List[float]] = None,
            training_time: float = 0.0):
        self.model_type = model_type
        self.params = params
        self.score = score
        self.model = model
        self.cv_scores = cv_scores if cv_scores is not None else []
        self.std_score = np.std(self.cv_scores) if self.cv_scores else 0.0
        self.training_time = training_time


class HyperparameterOptimizer:
    def __init__(self, seed: int, time_budget: int, n_trials: int, cv_folds: int = 5):
        self.seed = seed
        self.time_budget = time_budget
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.meta_features: Dict[str, Any] = {}
        self.start_time = None

        # DEHB parameters
        self.min_budget = 0.1
        self.max_budget = 1.0
        self.eta = 3

    def _get_configuration_space(self, model_type: str) -> ConfigurationSpace:
        """Create ConfigSpace configuration space for given model type."""
        if not self.meta_features:
            self.meta_features = {'n_samples': 1000, 'n_features': 10}

        n_samples = self.meta_features.get('n_samples', 1000)
        n_features = self.meta_features.get('n_features', 10)

        cs = ConfigurationSpace(seed=self.seed)

        if model_type == 'lightgbm':
            # Core parameters
            n_estimators = UniformIntegerHyperparameter(
                'n_estimators', 50, min(2000, max(100, n_samples // 2)), default_value=100
            )
            max_depth = UniformIntegerHyperparameter(
                'max_depth', 3, min(15, int(np.log2(max(n_features, 2))) + 5), default_value=6
            )
            learning_rate = UniformFloatHyperparameter(
                'learning_rate', 0.01, 0.3, default_value=0.1, log=True
            )
            num_leaves = UniformIntegerHyperparameter(
                'num_leaves', 10, min(300, 2**min(10, int(np.log2(max(n_features, 2))) + 3)), default_value=31
            )

            # Regularization parameters
            subsample = UniformFloatHyperparameter('subsample', 0.6, 1.0, default_value=1.0)
            colsample_bytree = UniformFloatHyperparameter('colsample_bytree', 0.6, 1.0, default_value=1.0)
            reg_alpha = UniformFloatHyperparameter('reg_alpha', 1e-8, 10.0, default_value=1e-8, log=True)
            reg_lambda = UniformFloatHyperparameter('reg_lambda', 1e-8, 10.0, default_value=1e-8, log=True)
            min_child_samples = UniformIntegerHyperparameter(
                'min_child_samples', 5, min(100, max(20, n_samples // 100)), default_value=20
            )

            cs.add([
                n_estimators, max_depth, learning_rate, num_leaves,
                subsample, colsample_bytree, reg_alpha, reg_lambda, min_child_samples
            ])

        elif model_type == 'xgboost':
            n_estimators = UniformIntegerHyperparameter(
                'n_estimators', 50, min(2000, max(100, n_samples // 2)), default_value=100
            )
            max_depth = UniformIntegerHyperparameter(
                'max_depth', 3, min(15, int(np.log2(max(n_features, 2))) + 5), default_value=6
            )
            learning_rate = UniformFloatHyperparameter(
                'learning_rate', 0.01, 0.3, default_value=0.1, log=True
            )
            subsample = UniformFloatHyperparameter('subsample', 0.6, 1.0, default_value=1.0)
            colsample_bytree = UniformFloatHyperparameter('colsample_bytree', 0.6, 1.0, default_value=1.0)
            reg_alpha = UniformFloatHyperparameter('reg_alpha', 1e-8, 10.0, default_value=1e-8, log=True)
            reg_lambda = UniformFloatHyperparameter('reg_lambda', 1e-8, 10.0, default_value=1e-8, log=True)
            min_child_weight = UniformIntegerHyperparameter(
                'min_child_weight', 1, min(20, max(5, n_samples // 100)), default_value=1
            )
            gamma = UniformFloatHyperparameter('gamma', 1e-8, 1.0, default_value=1e-8, log=True)

            cs.add([
                n_estimators, max_depth, learning_rate, subsample, colsample_bytree,
                reg_alpha, reg_lambda, min_child_weight, gamma
            ])

        elif model_type == 'catboost':
            iterations = UniformIntegerHyperparameter(
                'iterations', 50, min(2000, max(100, n_samples // 2)), default_value=100
            )
            depth = UniformIntegerHyperparameter(
                'depth', 3, min(10, int(np.log2(max(n_features, 2))) + 3), default_value=6
            )
            learning_rate = UniformFloatHyperparameter(
                'learning_rate', 0.01, 0.3, default_value=0.1, log=True
            )
            l2_leaf_reg = UniformFloatHyperparameter('l2_leaf_reg', 1e-8, 10.0, default_value=1e-8, log=True)
            subsample = UniformFloatHyperparameter('subsample', 0.6, 1.0, default_value=1.0)
            colsample_bylevel = UniformFloatHyperparameter('colsample_bylevel', 0.6, 1.0, default_value=1.0)
            min_data_in_leaf = UniformIntegerHyperparameter(
                'min_data_in_leaf', 1, min(50, max(10, n_samples // 100)), default_value=1
            )

            cs.add([
                iterations, depth, learning_rate, l2_leaf_reg,
                subsample, colsample_bylevel, min_data_in_leaf
            ])

        elif model_type == 'ridge':
            alpha_min = 1e-8 if n_samples > n_features else 1e-4
            alpha_max = 1000.0 if n_samples > n_features else 10000.0

            alpha = UniformFloatHyperparameter('alpha', alpha_min, alpha_max, default_value=1.0, log=True)
            fit_intercept = CategoricalHyperparameter('fit_intercept', [True, False], default_value=True)

            cs.add([alpha, fit_intercept])

        elif model_type == 'random_forest':
            n_estimators = UniformIntegerHyperparameter(
                'n_estimators', 50, min(500, max(100, n_samples // 2)), default_value=100
            )
            max_depth = UniformIntegerHyperparameter(
                'max_depth', 3, min(20, int(np.log2(max(n_features, 2))) + 8), default_value=10
            )
            min_samples_split = UniformIntegerHyperparameter(
                'min_samples_split', 2, min(20, max(5, n_samples // 100)), default_value=2
            )
            min_samples_leaf = UniformIntegerHyperparameter(
                'min_samples_leaf', 1, min(10, max(3, n_samples // 200)), default_value=1
            )
            max_features = UniformFloatHyperparameter('max_features', 0.3, 1.0, default_value=1.0)
            bootstrap = CategoricalHyperparameter('bootstrap', [True, False], default_value=True)

            cs.add([
                n_estimators, max_depth, min_samples_split,
                min_samples_leaf, max_features, bootstrap
            ])

        elif model_type == 'extra_trees':
            n_estimators = UniformIntegerHyperparameter(
                'n_estimators', 50, min(500, max(100, n_samples // 10)), default_value=100
            )
            max_depth = UniformIntegerHyperparameter(
                'max_depth', 3, min(20, int(np.log2(max(n_features, 2))) + 8), default_value=10
            )
            min_samples_split = UniformIntegerHyperparameter(
                'min_samples_split', 2, min(20, max(5, n_samples // 100)), default_value=2
            )
            min_samples_leaf = UniformIntegerHyperparameter(
                'min_samples_leaf', 1, min(10, max(3, n_samples // 200)), default_value=1
            )
            max_features = UniformFloatHyperparameter('max_features', 0.3, 1.0, default_value=1.0)
            bootstrap = CategoricalHyperparameter('bootstrap', [True, False], default_value=False)

            cs.add([
                n_estimators, max_depth, min_samples_split,
                min_samples_leaf, max_features, bootstrap
            ])

        elif model_type == 'elastic_net':
            alpha = UniformFloatHyperparameter('alpha', 1e-8, 100.0, default_value=1e-8, log=True)
            l1_ratio = UniformFloatHyperparameter('l1_ratio', 0.0, 1.0, default_value=0.5)

            cs.add([alpha, l1_ratio])

        return cs

    def _create_model(self, model_type: str, config: Configuration):
        """Create model instance with given configuration."""
        params = dict(config)

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
        """Get subset of data based on budget."""
        if budget >= 1.0:
            return X, y

        n_samples = int(len(X) * budget)
        n_samples = max(n_samples, 50)  # Minimum sample size

        indices = np.random.RandomState(self.seed).choice(len(X), size=n_samples, replace=False)
        return X.iloc[indices], y.iloc[indices]

    def _adjust_config_for_budget(self, config: Configuration, budget: float, model_type: str) -> Configuration:
        """Adjust configuration parameters based on budget for early stopping."""
        config_dict = dict(config)

        # Adjust tree-based model parameters based on budget for early stopping
        if model_type in ['lightgbm', 'xgboost'] and 'n_estimators' in config_dict:
            config_dict['n_estimators'] = max(50, int(config_dict['n_estimators'] * budget))
        elif model_type == 'catboost' and 'iterations' in config_dict:
            config_dict['iterations'] = max(50, int(config_dict['iterations'] * budget))
        elif model_type in ['random_forest', 'extra_trees'] and 'n_estimators' in config_dict:
            config_dict['n_estimators'] = max(50, int(config_dict['n_estimators'] * budget))

        return Configuration(config.config_space, config_dict)

    def _dehb_objective(self, config: Configuration, budget: float,
                        X: pd.DataFrame, y: pd.Series, model_type: str) -> Dict[str, float]:
        """DEHB objective function."""
        try:
            # Check time budget
            if self.start_time and (time.time() - self.start_time) > self.time_budget:
                return {'fitness': -np.inf, 'cost': budget}

            # Get budget subset
            X_subset, y_subset = self._get_budget_subset(X, y, budget)

            # Adjust configuration based on budget
            adjusted_config = self._adjust_config_for_budget(config, budget, model_type)

            # Create and evaluate model
            model = self._create_model(model_type, adjusted_config)

            # Use fewer CV folds for lower budgets to save time
            cv_folds = max(3, int(self.cv_folds * budget))
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.seed)

            scores = cross_val_score(model, X_subset, y_subset, cv=cv, scoring='r2', n_jobs=1)

            # Apply small penalty for lower budgets to prefer full evaluations
            score_penalty = 1.0 - (1.0 - budget) * 0.05
            fitness = np.mean(scores) * score_penalty

            return {'fitness': fitness, 'cost': budget}

        except Exception as e:
            logger.warning(f"DEHB evaluation failed for {model_type} with budget {budget:.2f}: {e}")
            return {'fitness': -np.inf, 'cost': budget}

    def _optimize_with_dehb(self, X: pd.DataFrame, y: pd.Series, model_type: str) -> ModelConfig:
        """Optimize using DEHB with ConfigSpace."""
        print("")
        logger.info(f"Optimizing {model_type} with DEHB and ConfigSpace...")

        # Get configuration space
        cs = self._get_configuration_space(model_type)
        if len(list(cs.values())) == 0:
            logger.warning(f"No hyperparameters defined for {model_type}")
            return self._get_default_config(model_type, X, y)

        # Initialize DEHB with ConfigSpace
        dehb_optimizer = DEHB(
            f=lambda config, fidelity, **kwargs: self._dehb_objective(config, fidelity, X, y, model_type),
            cs=cs,
            min_fidelity=self.min_budget,
            max_fidelity=self.max_budget,
            eta=self.eta,
            output_path="./data/output/tmp/dehb_runs/",
            n_workers=1,
            seed=self.seed
        )

        # Calculate budget allocation
        time_per_model = self.time_budget // 4  # Assuming 4 models on average
        max_iterations = max(20, self.n_trials // 4)  # At least 20 iterations per model

        model_start_time = time.time()

        try:
            # Run DEHB optimization
            traj, runtime, history = dehb_optimizer.run(
                # total_cost=self.time_budget,
                fevals=max_iterations,
                reset=True,
                seed=self.seed
            )

            logger.info(f"History: {history}")
            # logger.info(f"Trajectory: {traj}")
            # logger.info(f"Runtime: {runtime:.2f} seconds")

            # Get best configuration
            best_config, best_score = dehb_optimizer.get_incumbents()

            logger.info(f"{model_type} DEHB completed with score: {best_score:.4f}")

        except Exception as e:
            logger.warning(f"DEHB optimization failed for {model_type}: {e}")
            return self._get_default_config(model_type, X, y)

        # Train final model with best parameters on full data
        try:
            final_model = self._create_model(model_type, best_config)
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
            final_scores = cross_val_score(final_model, X, y, cv=cv, scoring='r2', n_jobs=1)
            final_score = np.mean(final_scores)

            # Fit the model on full data
            final_model.fit(X, y)

            total_time = time.time() - model_start_time

            logger.info(f"{model_type} final validation score: {final_score:.4f} ± {np.std(final_scores):.4f}")

            return ModelConfig(
                model_type=model_type,
                params=dict(best_config),
                score=final_score,
                model=final_model,
                cv_scores=final_scores.tolist(),
                training_time=total_time
            )

        except Exception as e:
            logger.error(f"Failed to train final model for {model_type}: {e}")
            return self._get_default_config(model_type, X, y)

    def _get_default_config(self, model_type: str, X: pd.DataFrame, y: pd.Series) -> ModelConfig:
        """Get default configuration for a model type."""
        cs = self._get_configuration_space(model_type)
        if len(list(cs.values())) == 0:
            # Fallback to hardcoded defaults
            default_params = {
                "lightgbm": {'n_estimators': 100, 'learning_rate': 0.1},
                "xgboost": {'n_estimators': 100, 'learning_rate': 0.1},
                "catboost": {'iterations': 100, 'learning_rate': 0.1},
                "ridge": {'alpha': 1.0},
                "random_forest": {'n_estimators': 100},
                "extra_trees": {'n_estimators': 100},
                "elastic_net": {'alpha': 1.0, 'l1_ratio': 0.5}
            }.get(model_type, {})
        else:
            # Use ConfigSpace default configuration
            default_config = cs.get_default_configuration()
            default_params = dict(default_config)

        try:
            if len(list(cs.values())) > 0:
                default_config = cs.get_default_configuration()
                model = self._create_model(model_type, default_config)
            else:
                # Create model with hardcoded params
                if model_type == 'lightgbm':
                    model = lgb.LGBMRegressor(random_state=self.seed, verbose=-1, **default_params)
                elif model_type == 'xgboost':
                    model = xgb.XGBRegressor(random_state=self.seed, verbosity=0, **default_params)
                elif model_type == 'catboost':
                    model = cb.CatBoostRegressor(random_state=self.seed, verbose=False, **default_params)
                elif model_type == 'ridge':
                    model = Ridge(random_state=self.seed, **default_params)
                else:
                    model = self._create_model(model_type, Configuration(cs, default_params))

            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=1)
            score = np.mean(scores)

            model.fit(X, y)

            logger.info(f"{model_type} using default parameters with score: {score:.4f}")

            return ModelConfig(
                model_type=model_type,
                params=default_params,
                score=score,
                model=model,
                cv_scores=scores.tolist(),
                training_time=0.0
            )
        except Exception as e:
            logger.error(f"Failed to create default config for {model_type}: {e}")
            raise

    def optimize_models(self, X: pd.DataFrame, y: pd.Series, model_types: List[str]) -> List[ModelConfig]:
        """Optimize models using DEHB with ConfigSpace."""
        self.start_time = time.time()
        self.meta_features = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'categorical_ratio': len(X.select_dtypes(exclude=[np.number]).columns) / X.shape[1],
            'numerical_ratio': len(X.select_dtypes(include=[np.number]).columns) / X.shape[1],
            'missing_ratio': X.isnull().sum().sum() / (X.shape[0] * X.shape[1]),
            'samples_to_features_ratio': X.shape[0] / X.shape[1] if X.shape[1] > 0 else 0,
        }

        logger.info(f"Starting DEHB optimization with ConfigSpace for {len(model_types)} models...")
        logger.info(f"Time budget: {self.time_budget} seconds")
        logger.info(f"Meta features: {self.meta_features}")

        optimized_models = []

        for model_type in model_types:
            try:
                elapsed_time = time.time() - self.start_time
                if elapsed_time > self.time_budget * 0.9:
                    logger.info("Approaching time budget limit, stopping optimization")
                    break

                model_config = self._optimize_with_dehb(X, y, model_type)
                optimized_models.append(model_config)

            except Exception as e:
                logger.error(f"Failed to optimize {model_type}: {e}")
                # Try to get default config as fallback
                try:
                    default_config = self._get_default_config(model_type, X, y)
                    optimized_models.append(default_config)
                except:
                    logger.error(f"Failed to create fallback config for {model_type}")
                    continue

        # Sort by score
        optimized_models.sort(key=lambda x: x.score, reverse=True)

        if optimized_models:
            best = optimized_models[0]
            logger.info(f"Best model: {best.model_type} with score {best.score:.4f} (±{best.std_score:.4f})")
        else:
            logger.warning("No models were successfully optimized.")

        total_time = time.time() - self.start_time
        logger.info(f"Total optimization time: {total_time:.2f} seconds")

        return optimized_models
