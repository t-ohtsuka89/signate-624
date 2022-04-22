import time

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset
from transformers.tokenization_utils import PreTrainedTokenizer


class BaseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, include_labels=True):
        self.df = df
        self.include_labels = include_labels

        if self.include_labels:
            self.labels = df["pm25_mid"].to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        target = self.df.loc[idx]
        date = (target["date"],)
        coordinate = target[["lat", "lon"]].astype(np.float32).to_numpy()
        variety = ["cnt", "min", "mid", "max", "var"]
        cols = ["co", "o3", "so2", "no2", "temperature", "humidity", "pressure", "ws", "dew"]
        res_dict = {}
        for col in cols:
            res_dict[col] = torch.FloatTensor(target[[col + "_" + v for v in variety]].astype(np.float32).to_numpy())
        res_dict["date"] = torch.FloatTensor(date)
        res_dict["coordinate"] = torch.FloatTensor(coordinate)

        if self.include_labels:
            res_dict["label"] = self.labels[idx]
            return res_dict

        return res_dict
