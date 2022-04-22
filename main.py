import os
import time
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn import preprocessing
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from torch.utils.data import DataLoader
from typing_extensions import Literal

from dataset import BaseDataset
from model import LightningModel
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


def train_model(
    model: pl.LightningModule, train: pd.DataFrame, fold: int, output_dir: str, epochs: int, device: torch.device
):

    trn_idx = train[train["fold"] != fold].index
    val_idx = train[train["fold"] == fold].index

    train_folds = train.loc[trn_idx].reset_index(drop=True)
    train_folds = train_folds.drop("fold", axis=1)
    valid_folds = train.loc[val_idx].reset_index(drop=True)
    valid_folds = valid_folds.drop("fold", axis=1)

    train_dataset = BaseDataset(train_folds)
    valid_dataset = BaseDataset(valid_folds)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    model.to(device)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir),
        filename=f"model_{fold:02d}",
        monitor="RMSE",
        verbose=False,
        save_last=False,
        save_top_k=1,
        save_weights_only=False,
        mode="min",
    )
    early_stopping = EarlyStopping(monitor="val_loss", mode="min")
    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=epochs,
        callbacks=[checkpoint_callback, early_stopping],
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


def main(hparams):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = hparams.data_dir
    output_dir = hparams.output_dir
    epochs = hparams.epochs
    fold_size = hparams.fold
    seed = hparams.seed
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    train = get_train_data(train, n_split=fold_size, seed=seed, fold_method=hparams.fold_method)
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))

    # date
    train = preprocess_date(train)
    test = preprocess_date(test)

    variety = ["min", "mid", "max", "var"]
    cols = ["co", "o3", "so2", "no2", "temperature", "humidity", "pressure", "ws", "dew"]
    ss = preprocessing.StandardScaler()
    mms = preprocessing.MinMaxScaler()
    # columns = [col + "_" + v for col in cols for v in variety]
    # train[columns] = ss.fit_transform(train[columns])
    columns = [col + "_cnt" for col in cols]
    columns.append("date")
    train[columns] = mms.fit_transform(train[columns])
    test[columns] = mms.transform(test[columns])
    sub = pd.read_csv(os.path.join(data_dir, "submit_sample.csv"), header=None)
    sub.columns = ["id", "judgement"]

    for fold in range(fold_size):
        model = LightningModel()
        train_model(model, train, fold, output_dir=output_dir, epochs=epochs, device=device)

    # test_data = torch.tensor(test.values.astype(np.float32))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--output_dir", default="outputs", type=str)
    parser.add_argument("--fold", default=5, type=int)
    parser.add_argument("--epochs", default=-1, type=int)
    parser.add_argument("--seed", default=472, type=int)
    parser.add_argument(
        "--fold_method", default="kfold", type=str, choices=["kfold", "stratified_fold", "group_kfold"]
    )
    args = parser.parse_args()
    main(args)
