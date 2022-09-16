import numpy as np
import pandas as pd


def get_float_columns(df: pd.DataFrame) -> list[str]:
    types = [np.float64, np.float32]

    columns = []
    for c in df.columns:
        if type(df[c].iloc[0]) in types:
            columns.append(c)

    return columns
