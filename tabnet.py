import os
import time
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn import preprocessing
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from torch.utils.data import DataLoader

from utils.fold import get_train_data


def to_unixtime(time_str: str) -> int:
    date_time = time.strptime(time_str, "%Y%M%d")
    unix_time = int(time.mktime(date_time))
    return unix_time


def preprocess_date(df: pd.DataFrame):
    df["year"] = df["year"].apply(lambda x: str(x).zfill(4))
    df["month"] = df["month"].apply(lambda x: str(x).zfill(2))
    df["day"] = df["day"].apply(lambda x: str(x).zfill(2))
    df["date"] = df["year"] + df["month"] + df["day"]
    df["date"] = df["date"].apply(to_unixtime)
    df.drop(columns=["year", "month", "day"], inplace=True)
    return df


def main(hparams):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = hparams.data_dir
    output_dir = hparams.output_dir
    epochs = hparams.epochs
    fold_size = hparams.fold
    seed = hparams.seed
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    train = get_train_data(train, n_split=fold_size, seed=seed)
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))

    # date
    train = preprocess_date(train)
    test = preprocess_date(test)

    variety = ["min", "mid", "max", "var"]
    cols = ["co", "o3", "so2", "no2", "temperature", "humidity", "pressure", "ws", "dew"]
    ss = preprocessing.StandardScaler()
    mms = preprocessing.MinMaxScaler()
    columns = [col + "_cnt" for col in cols]
    columns.append("date")
    train[columns] = mms.fit_transform(train[columns])
    test[columns] = mms.transform(test[columns])
    sub = pd.read_csv(os.path.join(data_dir, "submit_sample.csv"), header=None)
    sub.columns = ["id", "judgement"]

    for fold in range(fold_size):
        trn_idx = train[train["fold"] != fold].index
        val_idx = train[train["fold"] == fold].index

        train_folds = train.loc[trn_idx].reset_index(drop=True)
        train_folds = train_folds.drop("fold", axis=1)
        valid_folds = train.loc[val_idx].reset_index(drop=True)
        valid_folds = valid_folds.drop("fold", axis=1)

        variety = ["cnt", "min", "mid", "max", "var"]
        cols = ["co", "o3", "so2", "no2", "temperature", "humidity", "pressure", "ws", "dew"]
        features = [col + "_" + v for col in cols for v in variety]
        features.append("date")
        features.append("lat")
        features.append("lon")
        y_train = train_folds["pm25_mid"].to_numpy().reshape((-1, 1))
        y_valid = valid_folds["pm25_mid"].to_numpy().reshape((-1, 1))

        tabnet_params = dict(
            n_d=8,
            n_a=8,
            n_steps=3,
            gamma=1.3,
            n_independent=2,
            n_shared=2,
            seed=seed,
            lambda_sparse=1e-3,
            optimizer_fn=torch.optim.RAdam,
            optimizer_params=dict(lr=2e-2),
            mask_type="entmax",
            scheduler_params=dict(
                mode="min",
                patience=5,
                min_lr=1e-5,
                factor=0.9,
            ),
            verbose=10,
        )

        clf = TabNetRegressor(**tabnet_params)
        clf.fit(
            train_folds[features].to_numpy(),
            y_train,
            eval_set=[
                (
                    valid_folds[features].to_numpy(),
                    y_valid,
                )
            ],
            eval_metric=["rmse"],
        )
        clf.save_model(os.path.join(output_dir, f"tabnet-{fold:02d}"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--fold", default=5, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--seed", default=472, type=int)
    args = parser.parse_args()
    main(args)
