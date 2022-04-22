import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


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


def preprocess_coordinate(df: pd.DataFrame):
    # [lat, lon]
    min_li = [-90, -180]
    max_li = [90, 180]
    min_max_li = np.array([min_li, max_li])
    min_max_li = pd.DataFrame(min_max_li, columns=["lat", "lon"])

    mmscaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    mmscaler.fit(min_max_li.astype(np.float32))
    df[["lat", "lon"]] = mmscaler.transform(df[["lat", "lon"]])
    return df
