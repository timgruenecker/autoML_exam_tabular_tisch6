from sklearn.linear_model import Ridge
from sklearn.ensemble import VotingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
import logging
from src.automl.preprocessing import Preprocessor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, KFold



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
        meta = self.preprocessing.meta_features_
        self.logger.info(f"Meta-features used for HPO: {meta}")

        # Ridge Regression: Baseline-Modell mit Default-Werten
        ridge = Ridge(alpha=1.0)

        # Extract meta-features from Preprocessor
        meta = self.preprocessing.meta_features_
        logger.info(f"Meta-features extracted: {meta}")
        n_samples = meta.get("n_samples", X.shape[0])
        n_features = meta.get("n_features", X.shape[1])
        cardinality = meta.get("mean_cardinality", 10)

        # Dynamisch angepasste HPO-Suchräume basierend auf den Meta-Features
        max_estimators = min(500, int(n_samples / 2))
        max_depth_upper = min(16, int(cardinality + n_features / 2))

        # Optuna-Objective für LightGBM
        def objective_lgb(trial):
            n_features = meta["n_features"]
            n_samples = meta["n_samples"]

            max_estimators = 500 if n_samples > 5000 else 200
            max_depth_upper = 12 if n_features > 50 else 6

            return LGBMRegressor(
                n_estimators=trial.suggest_int("n_estimators", 50, max_estimators),
                max_depth=trial.suggest_int("max_depth", 3, max_depth_upper),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                random_state=self.seed,
            )

        # Optuna-Studie konfigurieren
        study_lgb = optuna.create_study(direction="maximize")
        study_lgb.enqueue_trial({
            "n_estimators": min(100, max_estimators),
            "max_depth": min(6, max_depth_upper),
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        })
        study_lgb.optimize(lambda trial: self._evaluate(objective_lgb(trial), X, y), n_trials=10)

        # Bestes LightGBM-Modell instanziieren
        lgb_model = objective_lgb(study_lgb.best_trial)
        self.logger.info(f"Best LightGBM trial: {study_lgb.best_trial.params}")

        # CatBoost: ohne HPO (Default mit Random Seed)
        cat_model = CatBoostRegressor(verbose=0, random_seed=self.seed)

        return {"ridge": ridge, "lightgbm": lgb_model, "catboost": cat_model}

    def _evaluate(self, model, X, y):
        """Evaluate model using KFold cross-validation and return mean R²."""
        cv = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)
        return scores.mean()

    def fit(self, X, y):
        logger.info("Fitting preprocessing...")
        X_transformed = self.preprocessing.fit_transform(X)
        logger.info("Fitting models...")
        self.models = self._build_models(X_transformed, y)

        for name, model in self.models.items():
            model.fit(X_transformed, y)
            preds = model.predict(X_transformed)
            score = r2_score(y, preds)
            logger.info(f"Model {name} R² on train: {score:.4f}")

        self.ensemble = VotingRegressor(estimators=[(k, m) for k, m in self.models.items()])
        print("Ensembled models:", self.ensemble.estimators)
        self.ensemble.fit(X_transformed, y)
        print("Final models:", list(self.models.keys()))


    def predict(self, X):
        X_transformed = self.preprocessing.transform(X)
        return self.ensemble.predict(X_transformed)
