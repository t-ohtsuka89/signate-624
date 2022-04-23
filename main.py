import os
from argparse import ArgumentParser
from typing import Dict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn import preprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BaseDataset
from model import LightningModel
from utils.fold import give_fold_index
from utils.preprocess import preprocess_date


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
    data_dir: str = hparams.data_dir
    output_dir: str = hparams.output_dir
    epochs: int = hparams.epochs
    fold_size: int = hparams.fold
    seed: int = hparams.seed
    os.makedirs(output_dir, exist_ok=True)
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    train = give_fold_index(train, n_split=fold_size, seed=seed, fold_method=hparams.fold_method)
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))

    # date
    train = preprocess_date(train)
    test = preprocess_date(test)

    cols = ["co", "o3", "so2", "no2", "temperature", "humidity", "pressure", "ws", "dew"]
    ss = preprocessing.StandardScaler()
    mms = preprocessing.MinMaxScaler()
    train[["pm25_mid"]] = ss.fit_transform(train[["pm25_mid"]])
    columns = [col + "_cnt" for col in cols]
    columns.append("date")
    mms.fit(pd.concat([train, test])[columns])
    train[columns] = mms.transform(train[columns])
    test[columns] = mms.transform(test[columns])
    sub = pd.read_csv(os.path.join(data_dir, "submit_sample.csv"), header=None)
    sub.columns = ["id", "judgement"]

    test_ds = BaseDataset(test, include_labels=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=16,
        shuffle=False,
        pin_memory=True,
    )
    predictions = []
    for fold in range(fold_size):
        model = LightningModel()
        train_model(model, train, fold, output_dir=output_dir, epochs=epochs, device=device)
        model.eval()
        preds = []
        for batch in tqdm(test_loader, total=len(test_loader)):
            batch: Dict[str, torch.Tensor]
            batch = {k: v.to(device, dtype=torch.float) for (k, v) in batch.items()}

            with torch.no_grad():
                y_preds: torch.Tensor = model.forward(**batch)
                y_preds = y_preds.float()
            preds.append(y_preds.cpu().numpy())
        preds = np.concatenate(preds)
        predictions.append(preds)
    predictions = np.mean(predictions, axis=0)
    predictions = ss.inverse_transform(predictions.reshape(-1, 1))
    sub = pd.read_csv(os.path.join(data_dir, "submit_sample.csv"), header=None)
    sub.columns = ["id", "judgement"]
    pd.Series(predictions).to_csv(os.path.join(output_dir, "predictions.csv"), index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--output_dir", default=os.path.join("outputs", "original"), type=str)
    parser.add_argument("--fold", default=5, type=int)
    parser.add_argument("--epochs", default=-1, type=int)
    parser.add_argument("--seed", default=472, type=int)
    parser.add_argument(
        "--fold_method", default="kfold", type=str, choices=["kfold", "stratified_fold", "group_kfold"]
    )
    args = parser.parse_args()
    main(args)
