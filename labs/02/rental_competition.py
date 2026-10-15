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

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")
parser.add_argument("--folds", default=5, type=int, help="Cross-validation folds")

# Several possible download locations (the year in the URL changes between course years).
DATA_URLS = [
    "https://ufal.mff.cuni.cz/~courses/npfl129/2627/datasets/",
    "https://ufal.mff.cuni.cz/~courses/npfl129/2526/datasets/",
    "https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/",
]


class Dataset:
    """Rental Dataset.

    The dataset instances consist of the following 12 features:
    - season (1: winter, 2: spring, 3: summer, 4: autumn)
    - year (0: 2011, 1: 2012)
    - month (1-12)
    - hour (0-23)
    - holiday (binary indicator)
    - day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - working day (binary indicator; a day is neither weekend nor holiday)
    - weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - relative humidity (0-1 range)
    - windspeed (normalized to 0-1 range)

    The target variable is the number of rented bikes in the given hour.
    """
    def __init__(self, name="rental_competition.train.npz", url=None):
        path = self._locate(name)

        if path is None:
            # Only reached when training and the file is not present locally.
            base = os.path.basename(name)
            urls = [url] if url else DATA_URLS
            for u in urls:
                try:
                    print("Downloading dataset {} from {}...".format(base, u), file=sys.stderr)
                    urllib.request.urlretrieve(u + base, filename="{}.tmp".format(base))
                    os.rename("{}.tmp".format(base), base)
                    path = base
                    break
                except Exception as e:  # noqa
                    print("  failed: {}".format(e), file=sys.stderr)
            if path is None:
                raise FileNotFoundError(
                    "Could not find or download '{}'. Download it manually (see the course page) "
                    "and put it next to this script or in the current directory.".format(base))

        # Load the dataset: arrays become attributes (`data`, and `target` if present).
        with np.load(path) as dataset:
            for key in dataset.files:
                setattr(self, key, dataset[key])

    @staticmethod
    def _locate(name):
        candidates = [name, os.path.basename(name)]
        if "__file__" in globals():
            candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(name)))
        for c in candidates:
            if os.path.isfile(c):
                return c
        return None


def build_pipeline() -> Pipeline:
    # Feature indices: 0 season, 1 year, 2 month, 3 hour, 4 holiday, 5 weekday,
    # 6 working day, 7 weather, 8 temp, 9 feeling temp, 10 humidity, 11 windspeed.
    categorical = [0, 2, 3, 5, 7]
    binary = [1, 4, 6]
    numeric = [8, 9, 10, 11]

    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("bin", "passthrough", binary),
        ("num", StandardScaler(), numeric),
    ], sparse_threshold=0)

    # Degree 2 gives interactions such as hour x working day or hour x temperature,
    # which are essential for this data (rental patterns differ a lot per hour/day type).
    return Pipeline([
        ("prep", prep),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("ridge", Ridge()),
    ])


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        cv = KFold(args.folds, shuffle=True, random_state=args.seed)
        alphas = np.logspace(-3, 2, 11)

        best_model, best_score = None, -np.inf
        # Compare predicting the raw target with predicting its square root.
        for name in ["identity", "sqrt"]:
            if name == "identity":
                estimator = build_pipeline()
                grid = {"ridge__alpha": alphas}
            else:
                estimator = TransformedTargetRegressor(build_pipeline(), func=np.sqrt, inverse_func=np.square)
                grid = {"regressor__ridge__alpha": alphas}

            search = GridSearchCV(estimator, grid, cv=cv, scoring="neg_root_mean_squared_error")
            search.fit(train.data, train.target)
            print("{}: CV RMSE {:.2f}, params {}".format(name, -search.best_score_, search.best_params_))

            if search.best_score_ > best_score:
                best_model, best_score = search.best_estimator_, search.best_score_

        print("Selected model CV RMSE: {:.2f}".format(-best_score))
        model = best_model  # already refit on the whole training set

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        predictions = np.maximum(model.predict(test.data), 0)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
