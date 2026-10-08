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

from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline


parser = argparse.ArgumentParser()

parser.add_argument("--predict", default=None, type=str,
                    help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true",
                    help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int,
                    help="Random seed")
parser.add_argument("--model_path", default="rental_competition.model",
                    type=str, help="Model path")


class Dataset:
    """Rental Dataset."""

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

        dataset = np.load(name)

        for key, value in dataset.items():
            setattr(self, key, value)


def create_features(data):
    # I am copying the original features first.
    x = data.astype(np.float64).copy()

    hour = data[:, 3]
    month = data[:, 2]
    day = data[:, 5]

    # I am adding cyclic features because 23:00 and 00:00
    # are actually next to each other.
    x = np.column_stack([
        x,
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12),
        np.sin(2 * np.pi * day / 7),
        np.cos(2 * np.pi * day / 7),
    ])

    return x


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # I am training the model here.
        np.random.seed(args.seed)

        train = Dataset()

        x = create_features(train.data)
        y = train.target

        # These variables are categorical rather than continuous.
        categorical_columns = [
            0,  # season
            1,  # year
            2,  # month
            3,  # hour
            4,  # holiday
            5,  # day of week
            6,  # working day
            7,  # weather
        ]

        # The other columns are numerical.
        numerical_columns = [
            i for i in range(x.shape[1])
            if i not in categorical_columns
        ]

        # I am using one-hot encoding for the categorical variables.
        categorical = OneHotEncoder(
            handle_unknown="ignore"
        )

        preprocessor = ColumnTransformer([
            ("categorical", categorical, categorical_columns),
            ("numerical", "passthrough", numerical_columns)
        ])

        # I am using Ridge because it is one of the allowed linear models.
        # PolynomialFeatures gives the model a little more flexibility.
        model = make_pipeline(
            preprocessor,
            PolynomialFeatures(
                degree=2,
                include_bias=False
            ),
            Ridge(alpha=10.0)
        )

        model.fit(x, y)

        # I am saving the complete pipeline so that the same
        # preprocessing is automatically used during prediction.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # I am loading the model that was trained before.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        x = create_features(test.data)

        predictions = model.predict(x)

        # The number of bikes cannot be negative.
        predictions = np.maximum(predictions, 0)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args(
        [] if "__file__" not in globals() else None
    )
    main(main_args)
