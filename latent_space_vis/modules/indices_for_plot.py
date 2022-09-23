import numpy as np
import pandas as pd
import streamlit as st


def indices_for_plot_module(indices: np.ndarray, container: st.container = None) \
        -> (np.ndarray, pd.DataFrame):
    if container is None:
        container = st.container()

    col1, col2 = container.columns(2)
    n_points = col1.number_input('Number of points to plot', min_value=10, max_value=len(indices),
                                 value=2000 if len(indices) > 2000 else len(indices))
    random = col2.checkbox('Random indices?', value=True)
    if random:
        new_indices = np.random.choice(np.arange(len(indices)), size=n_points)
    else:
        new_indices = np.arange(0, n_points)

    return indices[new_indices]


@st.cache
def select_subset(label_df: pd.DataFrame, selections: dict[str, str]) -> np.ndarray:
    """Creates multiselect widgets for each label and returns indices of the subset of the data according to the
    selection. """
    columns = label_df.columns.to_list()

    for key, value in selections.items():
        if not value:
            # Nothing selected
            continue
        else:
            # indices = label_df[label_df[key].isin(value)].index.values
            label_df = label_df[label_df[key].isin(value)]

    return label_df.index
