import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn import preprocessing

from utils.fold import get_train_data
from utils.preprocess import preprocess_coordinate, preprocess_date


def main(hparams):
    data_dir = hparams.data_dir
    output_dir = hparams.output_dir
    fold_size = hparams.fold
    seed = hparams.seed
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    sub = pd.read_csv(os.path.join(data_dir, "submit_sample.csv"), header=None)
    sub.columns = ["id", "judgement"]
    train = get_train_data(train, n_split=fold_size, seed=seed, fold_method="group_kfold")

    # date
    train = preprocess_date(train)
    test = preprocess_date(test)

    # coordinate
    train = preprocess_coordinate(train)
    test = preprocess_coordinate(test)

    # City and Country
    countries = train["Country"].unique()
    train = pd.get_dummies(train, columns=["Country"])
    test = pd.get_dummies(test, columns=["Country"])

    variety = ["min", "mid", "max", "var"]
    cols = ["co", "o3", "so2", "no2", "temperature", "humidity", "pressure", "ws", "dew"]
    country_cols = [f"Country_{country}" for country in countries]
    columns = [col + "_cnt" for col in cols]
    columns.append("date")

    mms = preprocessing.MinMaxScaler()
    mms.fit(pd.concat([train, test])[columns])
    train[columns] = mms.transform(train[columns])
    test[columns] = mms.transform(test[columns])

    variety = ["co", "o3", "so2", "no2", "temperature", "humidity", "pressure", "ws", "dew"]
    metrics_list = ["min", "mid", "max", "var"]

    feature_cols = [v + "_" + metrics for v in variety for metrics in metrics_list]
    feature_cols.append("date")
    feature_cols.append("lat")
    feature_cols.append("lon")
    feature_cols.extend(country_cols)

    for fold in range(fold_size):
        trn_idx = train[train["fold"] != fold].index
        val_idx = train[train["fold"] == fold].index

        train_folds = train.loc[trn_idx].reset_index(drop=True)
        train_folds = train_folds.drop("fold", axis=1)
        valid_folds = train.loc[val_idx].reset_index(drop=True)
        valid_folds = valid_folds.drop("fold", axis=1)

        y_train = train_folds["pm25_mid"].to_numpy().reshape((-1, 1))
        y_valid = valid_folds["pm25_mid"].to_numpy().reshape((-1, 1))

        X_train = train_folds[feature_cols].to_numpy()
        X_valid = valid_folds[feature_cols].to_numpy()
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
            X_train,
            y_train,
            eval_set=[
                (
                    X_valid,
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
    parser.add_argument("--seed", default=472, type=int)
    args = parser.parse_args()
    main(args)
