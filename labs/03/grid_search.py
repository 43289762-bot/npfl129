#!/usr/bin/env python3
import argparse

import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> float:
    # Load the digits dataset.
    dataset = sklearn.datasets.load_digits()
    dataset.target = dataset.target % 2

    # If you want to learn about the dataset, you can print some information
    # about it using `print(dataset.DESCR)`.

    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        dataset.data, dataset.target, test_size=args.test_size, random_state=args.seed)

    pipeline = sklearn.pipeline.Pipeline([
        ("scaler", sklearn.preprocessing.MinMaxScaler()),
        ("polynomial", sklearn.preprocessing.PolynomialFeatures()),
        ("lr", sklearn.linear_model.LogisticRegression(random_state=args.seed)),
    ])

    param_grid = {
        "polynomial__degree": [1, 2],
        "lr__C": [0.01, 1, 100],
        "lr__solver": ["lbfgs", "sag"],
    }

    model = sklearn.model_selection.GridSearchCV(
        pipeline, param_grid, cv=sklearn.model_selection.StratifiedKFold(5))
    model.fit(train_data, train_target)

    test_accuracy = model.score(test_data, test_target)

    return 100 * test_accuracy


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(main_args)
    print("Test accuracy: {:.2f}%".format(test_accuracy))
