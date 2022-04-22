import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import norm


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
    data_dir = "data"
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    train_df = preprocess_date(train_df)
    test_df = preprocess_date(test_df)

    train_df.drop("country", inplace=True)
    test_df.drop("country", inplace=True)


if __name__ == "__main__":
    main()
