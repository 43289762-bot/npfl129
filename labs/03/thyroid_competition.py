#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="thyroid_competition.model", type=str, help="Model path")


class Dataset:
    """Thyroid Dataset.

    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features

    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """
    def __init__(self,
                 name="thyroid_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2526/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

                
        import sklearn.compose
        import sklearn.linear_model
        import sklearn.model_selection
        import sklearn.pipeline
        import sklearn.preprocessing

        # The first 15 features are binary, the last 6 are real-valued.
        preprocessing = sklearn.compose.ColumnTransformer([
            ("real", sklearn.preprocessing.StandardScaler(), list(range(15, 21))),
        ], remainder="passthrough")

        pipeline = sklearn.pipeline.Pipeline([
            ("preprocessing", preprocessing),
            ("polynomial", sklearn.preprocessing.PolynomialFeatures(include_bias=False)),
            ("scaler", sklearn.preprocessing.StandardScaler()),
            ("lr", sklearn.linear_model.LogisticRegression(max_iter=2000, random_state=args.seed)),
        ])

        param_grid = {
            "polynomial__degree": [1, 2, 3],
            "lr__C": [0.01, 0.1, 1, 10, 100],
        }

        search = sklearn.model_selection.GridSearchCV(
            pipeline, param_grid, cv=sklearn.model_selection.StratifiedKFold(5, shuffle=True, random_state=args.seed))
        search.fit(train.data, train.target)
        print("Best params: {}, cross-val accuracy: {:.2f}%".format(search.best_params_, 100 * search.best_score_))

        # Keep only the refitted best pipeline (smaller to serialize).
        model = search.best_estimator_

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

                
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
