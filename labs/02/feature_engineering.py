#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--dataset", default="diabetes", type=str, help="Standard sklearn dataset to load")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    dataset = getattr(sklearn.datasets, "load_{}".format(args.dataset))()

    # I am splitting the dataset into train and test using the given seed and test_size
    train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(dataset.data, dataset.target,test_size=args.test_size,random_state=args.seed)

    # I am figuring out which columns are categorical (all integer values)
    categorical = []
    numeric = []
    for i in range(dataset.data.shape[1]):
        col = dataset.data[:, i]
        # If all values are integers, I treat the column as categorical
        if np.all(col.astype(int) == col):
            categorical.append(i)
        else:
            numeric.append(i)

    # I am creating a ColumnTransformer that:
    # - one-hot encodes categorical columns
    # - standardizes numeric columns
    preprocess = sklearn.compose.ColumnTransformer(
        transformers=[
            ("cat", sklearn.preprocessing.OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical),("num", sklearn.preprocessing.StandardScaler(), numeric),])

    # I am adding polynomial features of degree 2 (without the bias column)
    poly = sklearn.preprocessing.PolynomialFeatures(degree=2,include_bias=False)

    # I am chaining everything together in a Pipeline
    pipeline = sklearn.pipeline.Pipeline([("preprocess", preprocess),("poly", poly),])

    # I am fitting the pipeline on the training data and transforming it
    train_data = pipeline.fit_transform(train_x)

    # I am transforming the test data using the already fitted pipeline
    test_data = pipeline.transform(test_x)

    # I return only the first 5 rows of each, as required
    return train_data[:5], test_data[:5]


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    train_data, test_data = main(main_args)
    for dataset in [train_data, test_data]:
        for line in range(min(dataset.shape[0], 5)):
            print(" ".join("{:.4g}".format(dataset[line, column]) for column in range(min(dataset.shape[1], 140))),
                  *["..."] if dataset.shape[1] > 140 else [])
