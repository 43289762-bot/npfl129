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

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge


parser = argparse.ArgumentParser()

# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str,
                    help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true",
                    help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int,
                    help="Random seed")

parser.add_argument("--model_path", default="rental_competition.model",
                    type=str, help="Model path")


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

    def __init__(
        self,
        name="rental_competition.train.npz",
        url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"
    ):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(
                url + name,
                filename="{}.tmp".format(name)
            )
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


def make_features(data):
    """
    I am adding some extra features here because hour and month are
    not really continuous variables in the way a normal regression sees them.
    """

    x = np.asarray(data)

    # The original data has 12 columns.
    # I keep the original features and add a few useful ones.
    result = x.astype(np.float64).copy()

    hour = x[:, 3]
    month = x[:, 2]
    day = x[:, 5]

    # I am adding cyclic versions of hour and month.
    # This makes 23:00 and 00:00 close to each other.
    result = np.column_stack([
        result,
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12),
        np.sin(2 * np.pi * day / 7),
        np.cos(2 * np.pi * day / 7),
    ])

    # I am also adding a rough hour-of-week feature.
    # The one-hot encoder later can use this as a category.
    hour_of_week = day * 24 + hour

    result = np.column_stack([
        result,
        hour_of_week
    ])

    return result


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)

        train = Dataset()

        x = make_features(train.data)
        y = train.target

        # These are the columns that are really categories.
        # I don't want the regression to assume that, for example,
        # month 12 is twelve times month 1.
        categorical_columns = [
            0,   # season
            1,   # year
            2,   # month
            3,   # hour
            4,   # holiday
            5,   # day of week
            6,   # working day
            7,   # weather
            19   # hour of week
        ]

        # The remaining columns are numerical.
        all_columns = set(range(x.shape[1]))
        numerical_columns = sorted(all_columns - set(categorical_columns))

        # I am using one-hot encoding for the categorical variables.
        categorical_transformer = OneHotEncoder(
            handle_unknown="ignore"
        )

        # For numerical variables I am using splines.
        # This lets the linear model learn curves instead of only straight lines.
        numerical_transformer = make_pipeline(
            StandardScaler(),
            SplineTransformer(
                n_knots=8,
                degree=3,
                include_bias=False
            )
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    categorical_transformer,
                    categorical_columns
                ),
                (
                    "numerical",
                    numerical_transformer,
                    numerical_columns
                )
            ]
        )

        # Ridge is still a linear model, so it is allowed by the assignment.
        # The preprocessing gives it enough flexibility for this dataset.
        model = make_pipeline(
            preprocessor,
            Ridge(alpha=10.0)
        )

        # I am fitting everything on the complete training set.
        model.fit(x, y)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # We are predicting the test set.
        test = Dataset(args.predict)

        x = make_features(test.data)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # I am using exactly the same feature processing as during training.
        predictions = model.predict(x)

        # Bike rentals cannot be negative, so I clip these predictions.
        predictions = np.maximum(predictions, 0)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args(
        [] if "__file__" not in globals() else None
    )
    main(main_args)
