import numpy as np
import pandas as pd


def rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def chia_train_val_test(df: pd.DataFrame, cot_gia_tri='value', ratios=(0.7, 0.15, 0.15)):
    """Chia dataframe theo thời gian: train/val/test theo ratios (tổng =1)."""
    n = len(df)
    r1 = int(n * ratios[0])
    r2 = r1 + int(n * ratios[1])
    df_train = df.iloc[:r1]
    df_val = df.iloc[r1:r2]
    df_test = df.iloc[r2:]
    return df_train, df_val, df_test
