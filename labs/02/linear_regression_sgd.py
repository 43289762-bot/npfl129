#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--epochs", default=50, type=int, help="Number of SGD training epochs")
parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization strength")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=92, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[list[float], float, float]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial regression dataset.
    data, target = sklearn.datasets.make_regression(n_samples=args.data_size, random_state=args.seed)

    #I am appending the bias
    data = np.concatenate([data, np.ones((data.shape[0], 1))], axis=1)


    #I am splitting the test data and the training data
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(data, target, test_size=args.test_size, random_state=args.seed)


    # I am initialazing the weights
    weights = generator.uniform(size=train_data.shape[1], low=-0.1, high=0.1)

    #We are iterating every epoch
    train_rmses, test_rmses = [], []
    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])
        
        for start in range(0, train_data.shape[0], args.batch_size):
            batch_idx = permutation[start:start + args.batch_size]
            Xb = train_data[batch_idx]
            tb = train_target[batch_idx]
            
            preds = Xb @ weights
            grad = (preds - tb)[:, None] * Xb
            grad = grad.mean(axis=0) 
            
            reg = args.l2 * np.copy(weights)
            reg[-1] = 0
            grad += reg   
            
            weights -= args.learning_rate * grad 
        
        #I am computing the RMSE
        train_rmses.append(np.sqrt(sklearn.metrics.mean_squared_error(train_target, train_data @ weights)))
        test_rmses.append(np.sqrt(sklearn.metrics.mean_squared_error(test_target, test_data @ weights)))

    
    #I am training the model
    explicit_model = sklearn.linear_model.LinearRegression().fit(train_data, train_target)
    explicit_rmse = np.sqrt(sklearn.metrics.mean_squared_error(test_target, explicit_model.predict(test_data)))


    

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(train_rmses, label="Train")
        plt.plot(test_rmses, label="Test")
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")
        plt.legend()
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return weights, test_rmses[-1], explicit_rmse


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, sgd_rmse, explicit_rmse = main(main_args)
    print("Test RMSE: SGD {:.3f}, explicit {:.1f}".format(sgd_rmse, explicit_rmse))
    print("Learned weights:", *("{:.3f}".format(weight) for weight in weights[:12]), "...")
