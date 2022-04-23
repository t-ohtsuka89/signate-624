import os

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.trial import Trial
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn import preprocessing

from utils.fold import give_fold_index
from utils.preprocess import preprocess_coordinate, preprocess_date

# const
DATA_DIR = "data"
OUTPUT_DIR = os.path.join("outputs", "tabnet-optuna")
SEED = 2022
FOLD_SIZE = 5

# load data
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# date
train_df = preprocess_date(train_df)
test_df = preprocess_date(test_df)

# coordinate
train_df = preprocess_coordinate(train_df)
test_df = preprocess_coordinate(test_df)

# City and Country
countries = train_df["Country"].unique()
train_df = pd.get_dummies(train_df, columns=["Country"])
test_df = pd.get_dummies(test_df, columns=["Country"])


# scaling
variety = ["min", "mid", "max", "var"]
cols = ["co", "o3", "so2", "no2", "temperature", "humidity", "pressure", "ws", "dew"]
country_cols = [f"Country_{country}" for country in countries]
columns = [col + "_cnt" for col in cols]
columns.append("date")
mms = preprocessing.MinMaxScaler()
ss = preprocessing.StandardScaler()
mms.fit(pd.concat([train_df, test_df])[columns])
train_df[["pm25_mid"]] = ss.fit_transform(train_df[["pm25_mid"]])
train_df[columns] = mms.transform(train_df[columns])
test_df[columns] = mms.transform(test_df[columns])

# fold
train_df = give_fold_index(train_df, n_split=FOLD_SIZE, seed=SEED, fold_method="group_kfold")

# feature columns
variety = ["co", "o3", "so2", "no2", "temperature", "humidity", "pressure", "ws", "dew"]
metrics_list = ["min", "mid", "max", "var"]
feature_cols = [v + "_" + metrics for v in variety for metrics in metrics_list]
feature_cols.append("date")
feature_cols.append("lat")
feature_cols.append("lon")
feature_cols.extend(country_cols)


def Objective(trial: Trial):
    mask_type = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
    n_da = trial.suggest_int("n_da", 8, 64, step=8)
    n_steps = trial.suggest_int("n_steps", 1, 10, step=3)
    gamma = trial.suggest_float("gamma", 1.0, 2.0, step=0.2)
    n_shared = trial.suggest_int("n_shared", 1, 3)
    tabnet_params = dict(
        n_d=n_da,
        n_a=n_da,
        n_steps=n_steps,
        gamma=gamma,
        lambda_sparse=1e-3,
        optimizer_fn=torch.optim.RAdam,
        optimizer_params=dict(lr=1e-3),
        mask_type=mask_type,
        n_shared=n_shared,
        scheduler_params=dict(
            mode="min",
            patience=trial.suggest_int("patienceScheduler", low=3, high=10),
            min_lr=1e-5,
            factor=0.5,
        ),
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        verbose=0,
    )

    CV_score_array = []
    for fold in range(FOLD_SIZE):
        trn_idx = train_df[train_df["fold"] != fold].index
        val_idx = train_df[train_df["fold"] == fold].index

        train_folds = train_df.loc[trn_idx].copy().reset_index(drop=True)
        train_folds = train_folds.drop("fold", axis=1)
        valid_folds = train_df.loc[val_idx].copy().reset_index(drop=True)
        valid_folds = valid_folds.drop("fold", axis=1)

        y_train = train_folds["pm25_mid"].to_numpy().reshape((-1, 1))
        y_valid = valid_folds["pm25_mid"].to_numpy().reshape((-1, 1))

        X_train = train_folds[feature_cols].to_numpy()
        X_valid = valid_folds[feature_cols].to_numpy()

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
            patience=trial.suggest_int("patience", low=5, high=30),
            max_epochs=trial.suggest_int("epochs", 1, 100),
        )
        CV_score_array.append(clf.best_cost)  # type: ignore
    avg = np.mean(CV_score_array)
    return avg


import pickle

study = optuna.create_study(direction="minimize", study_name="TabNet optimization")
study.optimize(Objective, timeout=6 * 60)
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "params.pkl"), "wb") as f:
    pickle.dump(study.best_params, f)
