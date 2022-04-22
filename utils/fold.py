from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from typing_extensions import Literal


def get_train_data(
    train: pd.DataFrame,
    n_split=5,
    seed=42,
    fold_method: Literal["kfold", "stratified_fold", "group_kfold"] = "kfold",
):
    if fold_method == "kfold":
        kf = KFold(n_splits=n_split, shuffle=True, random_state=seed)
        for n, (_, val_index) in enumerate(kf.split(train, train["pm25_mid"])):
            train.loc[val_index, "fold"] = int(n)
    elif fold_method == "stratified_fold":
        df = pd.DataFrame({"target": train["pm25_mid"]})
        bins = list(range(0, 500, 10))
        df["bins"] = pd.cut(train["pm25_mid"], bins=len(bins), labels=False)
        skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=seed)
        for n, (_, val_index) in enumerate(skf.split(train, df["bins"])):
            train.loc[val_index, "fold"] = int(n)
    elif fold_method == "group_kfold":
        gkf = GroupKFold(n_splits=n_split)
        for n, (_, val_index) in enumerate(gkf.split(train, train["pm25_mid"], groups=train["City"])):
            train.loc[val_index, "fold"] = int(n)
    train["fold"] = train["fold"].astype(np.uint8)

    return train
