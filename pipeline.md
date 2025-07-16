1. Preprocessing
    - Fix missing values (e.g. median)
    - Encode categorial features as one-hot
    - Scaling numerical features (nur wenn sinnvoll für model)

2. Automatic Feature Engineering
    - Meta feature extraction (n_samples, n_features, ...)
    - Polynomial features
    - Frequency/target encoding
    - Feature Union -> stack engineered features with original ones

3. Feature Selection
    - Importance Filtering (e.g. LightGBM)
    - Remove constant/low variance features
    - Optional: PCA

4. Model Selection
    - LightGBM
    - CatBoost for categorial
    - Ridge (linear baseline)

5. HPO with early stopping
    - Meta-Learning: Use meta features for initial HPO ranges
    - Search space (tree depth, learning rate, regularization)
    - Optuna with pruning and bayesian search

6. Postprocessing
    - Model ensembling -> average top-k configs