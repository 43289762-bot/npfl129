#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=13, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float]:
    # I am loading the diabetes dataset.
    dataset = sklearn.datasets.load_diabetes()

    #I am splitting the training data and the test data
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(dataset.data, dataset.target, test_size=args.test_size, random_state=args.seed)
    
    #I am generating the lambdas 
    lambdas = np.geomspace(0.01, 10, num=500)

    
    #I am keeping and inicializing the best lambda value
    best_lambda = None
    best_rmse = float("inf")
    rmses = []

    #I am training the model
    for lam in lambdas:
        model = sklearn.linear_model.Ridge(alpha=lam)
        model.fit(train_data, train_target)

        preds = model.predict(test_data)
        mse = sklearn.metrics.mean_squared_error(test_target, preds)
        rmse = np.sqrt(mse)
        rmses.append(rmse)

        if rmse < best_rmse:
            best_rmse = rmse
            best_lambda = lam
    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(lambdas, rmses)
        plt.xscale("log")
        plt.xlabel("L2 regularization strength $\\lambda$")
        plt.ylabel("RMSE")
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return best_lambda, best_rmse


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    best_lambda, best_rmse = main(main_args)
    print("{:.2f} {:.2f}".format(best_lambda, best_rmse))
