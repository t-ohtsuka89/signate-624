import os
import time
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BaseDataset
from model import LightningModel


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_size = 5

    data_dir = "data"
    output_dir = "outputs"
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))

    # date
    train = preprocess_date(train)
    test = preprocess_date(test)

    cols = ["co", "o3", "so2", "no2", "temperature", "humidity", "pressure", "ws", "dew"]
    mms = preprocessing.MinMaxScaler()
    columns = [col + "_cnt" for col in cols]
    columns.append("date")
    train[columns] = mms.fit_transform(train[columns])
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
        model = LightningModel.load_from_checkpoint(os.path.join("outputs", f"model_{fold:02}-v1.ckpt"))
        model.to(device)
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
    sub = pd.read_csv(os.path.join(data_dir, "submit_sample.csv"), header=None)
    sub.columns = ["id", "judgement"]
    pd.Series(predictions).to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    sub["judgement"] = predictions
    sub.to_csv(os.path.join("outputs", "submission.csv"), index=False, header=False)


if __name__ == "__main__":
    main()
