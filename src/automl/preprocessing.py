import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path: str) -> pd.DataFrame:
    """Loads CSV data from given path into a pandas DataFrame."""
    return pd.read_csv(path)

def preprocess_data(df: pd.DataFrame, target_column: str = "target"):
    """
    Splits dataframe into features and target. Assumes target_column exists in df.
    Returns: (X, y)
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y
