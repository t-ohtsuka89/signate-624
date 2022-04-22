import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn import preprocessing

from utils.preprocess import preprocess_date


def preprocess_mms(X, mms: preprocessing.MinMaxScaler):
    return mms.transform(X)


def main(hparams):
    data_dir = hparams.data_dir
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))

    train = preprocess_date(train)
    test = preprocess_date(test)

    cols = ["co", "o3", "so2", "no2", "temperature", "humidity", "pressure", "ws", "dew"]
    columns = [col + "_cnt" for col in cols]
    columns.append("date")
    mms = preprocessing.MinMaxScaler()
    mms.fit(pd.concat([train[columns], test[columns]]))
    train[columns] = preprocess_mms(train[columns], mms)
    test[columns] = preprocess_mms(test[columns], mms)

    sub = pd.read_csv(os.path.join(data_dir, "submit_sample.csv"), header=None)
    sub.columns = ["id", "judgement"]

    variety = ["cnt", "min", "mid", "max", "var"]
    cols = ["co", "o3", "so2", "no2", "temperature", "humidity", "pressure", "ws", "dew"]
    features = [col + "_" + v for col in cols for v in variety]
    features.append("date")
    features.append("lat")
    features.append("lon")

    preds = []
    for fold in range(hparams.fold):
        clf = TabNetRegressor()
        clf.load_model(os.path.join(hparams.output_dir, f"tabnet-{fold:02d}.zip"))
        tmp_preds: np.ndarray = clf.predict(test[features].to_numpy())
        preds.append(tmp_preds.reshape((-1,)))
    preds = np.mean(preds, axis=0)
    sub = pd.read_csv(os.path.join(data_dir, "submit_sample.csv"), header=None)
    sub.columns = ["id", "judgement"]

    sub["judgement"] = preds
    sub.to_csv(os.path.join("outputs", "submission.csv"), index=False, header=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--fold", default=5, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--seed", default=472, type=int)
    args = parser.parse_args()
    main(args)
