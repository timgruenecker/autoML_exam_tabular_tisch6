import argparse
import logging
from src.automl.evaluate import evaluate_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all models on a given dataset")
    parser.add_argument("--train", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--test", type=str, help="Optional: path to test CSV file")
    parser.add_argument("--target", type=str, default="target", help="Target column name")
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--output", type=str, default="outputs", help="Directory to save outputs")
    parser.add_argument("--quiet", action="store_true", help="Suppress logs")

    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO)

    evaluate_all(
        train_path=args.train,
        test_path=args.test,
        target_column=args.target,
        output_dir=args.output,
        cv=args.cv
    )
