"""AutoML class for regression tasks.

This module contains an example AutoML class that simply returns dummy predictions.
You do not need to use this setup or sklearn and you can modify this however you like.
"""
from __future__ import annotations

from typing import Protocol, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.dummy import DummyRegressor
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("automl")

METRICS = {"r2": r2_score}


class Model(Protocol):
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...


class Pipeline:

    def __init__(self, seed: int, metric: str, time_budget: int = 3600, n_trials: int = 100, topk: int = 2):
        self.seed = seed
        self.metric = metric
        self.time_budget = time_budget
        self.n_trials = n_trials
        self.topk = topk

        from .preprocessing import Preprocessor
        from .feature_engineering import FeatureEngineer
        from .feature_selection import FeatureSelector
        from .model_selection import ModelSelector
        from .hpo import HyperparameterOptimizer

        self.preprocessor = Preprocessor(seed=seed)
        self.feature_engineer = FeatureEngineer(seed=seed)
        self.feature_selector = FeatureSelector(seed=seed)
        self.model_selector = ModelSelector(seed=seed)
        self.hpo = HyperparameterOptimizer(seed=seed, time_budget=time_budget, n_trials=n_trials)

    def preprocess_and_split(self, X: pd.DataFrame, Y: pd.Series, val_set_size=0.2):
        logger.info("Applying Preprocessing to entire dataset...")

        logger.info("Step 1: Preprocessing")
        X_prepr = self.preprocessor.preprocess(X)

        logger.info("Step 2: Feature Engineering")
        X_eng = self.feature_engineer.fit(X_prepr, Y)

        logger.info("Step 3: Feature Selection")
        X_sel = self.feature_selector.select_features(X_eng, Y)

        return train_test_split(
            X_sel,
            Y,
            random_state=self.seed,
            test_size=val_set_size,
        )

    def apply_preprocessing(self, X: pd.DataFrame):
        X_prepr = self.preprocessor.transform(X)
        X_eng = self.feature_engineer.transform(X_prepr)
        return self.feature_selector.transform(X_eng)

    def run(self, X: pd.DataFrame, Y) -> Model:
        logger.info("Starting AutoML pipeline ...")

        # X = self.preprocessor.preprocess(X)
        # X = self.feature_engineer.fit(X, Y)
        # X = self.feature_selector.select_features(X, Y)

        logger.info("Step 4: Model Selection")
        meta_features = self.feature_engineer.get_meta_features()
        model_candidates = self.model_selector.select_models(meta_features)

        logger.info("Step 5: HPO with early stopping")
        optimized_models = self.hpo.optimize_models(X, Y, model_candidates)

        logger.info(f"HPO resulted in {len(optimized_models)} optimized models.")

        logger.info("Step 6: Postprocessing")
        from .postprocessing import EnsembleModel
        ensemble = EnsembleModel(
            models=optimized_models[:self.topk],
            preprocessor=self.preprocessor,
            feature_engineer=self.feature_engineer,
            feature_selector=self.feature_selector
        )
        return ensemble


class AutoML:

    def __init__(self, seed: int, metric: str = "r2", time_budget: int = 3600, n_trials: int = 100) -> None:
        self.seed = seed
        self.metric = METRICS[metric]
        self.pipeline = Pipeline(seed, metric, time_budget, n_trials, topk=1)
        self._model: Model | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> AutoML:

        X_train, X_val, y_train, y_val = self.pipeline.preprocess_and_split(X, y)

        self._model = self.pipeline.run(X_train, y_train)

        logger.info("Succesfully ran AutoML pipeline!")

        val_preds = self._model.predict(X_val)
        val_score = self.metric(y_val, val_preds)
        logger.info(f"Validation score: {val_score:.4f}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model not fitted")

        X_test = self.pipeline.apply_preprocessing(X)

        return self._model.predict(X_test)
