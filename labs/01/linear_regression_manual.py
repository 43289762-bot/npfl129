#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> float:
    # Load the diabetes dataset.
    dataset = sklearn.datasets.load_diabetes()

    #I am expanding X to append the bias as the last column
    X = dataset.data
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis = 1)
    

    #I am separating the training data from the teste data
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(X, dataset.target, test_size=args.test_size, random_state=args.seed)

    #I am applying the formula to calculate the weights
    w = np.linalg.inv(train_data.T @ train_data) @ (train_data.T @ train_target)

    #I am predicting target values on the test set.
    pred = test_data @ w

    #I am computing the RMSE
    rmse = np.sqrt(np.mean((pred - test_target)**2))


    return rmse


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    rmse = main(main_args)
    print("{:.2f}".format(rmse))


