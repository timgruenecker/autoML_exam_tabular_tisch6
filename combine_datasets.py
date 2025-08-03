import os
from pathlib import Path
import pandas as pd

# Root data directory
DATA_DIR = Path("./data")

# Expected filenames
FILE_NAMES = ["X_train.parquet", "X_test.parquet", "y_train.parquet", "y_test.parquet"]

# Iterate over each dataset in data/
for dataset_dir in DATA_DIR.iterdir():
    if dataset_dir.name == "output":
        continue
    if dataset_dir.is_dir():
        print(f"Processing dataset: {dataset_dir.name}")

        # Subdirectories 0-9
        subfolders = [dataset_dir / str(i) for i in range(1, 11)]

        # Output directory: 11
        output_dir = dataset_dir / "11"
        output_dir.mkdir(exist_ok=True)

        # For each of the four files, combine across subfolders
        for file_name in FILE_NAMES:
            combined_parts = []
            for subfolder in subfolders:
                file_path = subfolder / file_name
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    combined_parts.append(df)
                else:
                    print(f"Warning: Missing file {file_path}")

            # Concatenate and save
            combined_df = pd.concat(combined_parts, ignore_index=True)
            combined_df.to_parquet(output_dir / file_name)
            print(f"Saved: {output_dir / file_name}")
