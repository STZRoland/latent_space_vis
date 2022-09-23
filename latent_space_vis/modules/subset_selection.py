import numpy as np
import pandas as pd
import streamlit as st


def subset_selection_module(label_df: pd.DataFrame, container: st.container = None) -> np.ndarray:
    if container is None:
        container = st.container()
    container.header('Select a subset of the data')

    selections = {}
    for c in label_df.columns:
        options = label_df[c].unique()
        selections[c] = container.multiselect(c, options=options)

    return select_subset(label_df, selections)


@st.cache
def select_subset(label_df: pd.DataFrame, selections: dict[str, str]) -> np.ndarray:
    """Creates multiselect widgets for each label and returns indices of the subset of the data according to the
    selection. """

    for key, value in selections.items():
        if not value:
            # Nothing selected
            continue
        else:
            # indices = label_df[label_df[key].isin(value)].index.values
            label_df = label_df[label_df[key].isin(value)]

    return label_df.index.to_numpy()
