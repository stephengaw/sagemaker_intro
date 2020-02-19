"""
Building on the code from dummy_ml.py, refactor into a script to be run from the terminal.
"""
import os
import argparse
from utils import get_data, train_model, model_performance, save_model_artifacts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--penalty",
        type=str,
        default="l2",
    )

    parser.add_argument(
        "-C",
        type=float,
        default=1.0,
        help="Choose regularization strength.",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR"),
        help="Local directory to save the model",
    )

    parser.add_argument(
        "--train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN"),
        help="Local directory where training data lives",
    )

    return parser


if __name__ == "__main__":
    # get inputs
    parser = parse_args()
    args = parser.parse_args()

    # load data and split train and test sets
    x_train, x_test, y_train, y_test = get_data(args.train)

    # train model
    model = train_model(x_train, y_train, penalty=args.penalty, C=args.C)

    # model performance
    model_performance(model, x_test, y_test)

    # save model artifacts (model, metrics, graphs, etc)
    save_model_artifacts(model, args.model_dir)
