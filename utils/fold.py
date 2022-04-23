import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from typing_extensions import Literal


def give_fold_index(
    df: pd.DataFrame,
    n_split: int = 5,
    fold_method: Literal["kfold", "stratified_fold", "group_kfold"] = "kfold",
    seed=2022,
):
    if fold_method == "kfold":
        kf = KFold(n_splits=n_split, shuffle=True, random_state=seed)
        for n, (_, val_index) in enumerate(kf.split(df, df["pm25_mid"])):
            df.loc[val_index, "fold"] = int(n)
    elif fold_method == "stratified_fold":
        bins = list(range(0, 500, 10))
        df["bins"] = pd.cut(df["pm25_mid"], bins=len(bins), labels=False)
        skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=seed)
        for n, (_, val_index) in enumerate(skf.split(df, df["bins"])):
            df.loc[val_index, "fold"] = int(n)
        df.drop("bins", inplace=True)
    elif fold_method == "group_kfold":
        gkf = GroupKFold(n_splits=n_split)
        for n, (_, val_index) in enumerate(gkf.split(df, df["pm25_mid"], groups=df["City"])):
            df.loc[val_index, "fold"] = int(n)
    else:
        raise ValueError("invalid fold_method")
    df["fold"] = df["fold"].astype(np.uint8)
    return df
