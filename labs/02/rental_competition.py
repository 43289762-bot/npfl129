#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import numpy as np
import numpy.typing as npt

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--predict", default=None, type=str)
parser.add_argument("--recodex", default=False, action="store_true")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--model_path", default="rental_competition.model", type=str)

class Dataset:
    def __init__(self, name):
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)

def train_model(args):
    np.random.seed(args.seed)
    train = Dataset("rental_competition.train.npz")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("ridge", Ridge(alpha=1.0))
    ])

    model.fit(train.data, train.target)

    with lzma.open(args.model_path, "wb") as f:
        pickle.dump(model, f)

    return model

def main(args):
    # TRAINING MODE
    if args.predict is None:
        train_model(args)
        return None

    # PREDICTION MODE
    # If model does not exist, train it first
    if not os.path.exists(args.model_path):
        model = train_model(args)
    else:
        with lzma.open(args.model_path, "rb") as f:
            model = pickle.load(f)

    # Load test dataset
    test = Dataset(args.predict)
    return model.predict(test.data)

if __name__ == "__main__":
    main_args = parser.parse_args()
    main(main_args)
