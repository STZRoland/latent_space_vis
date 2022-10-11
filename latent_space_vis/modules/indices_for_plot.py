import numpy as np
import pandas as pd
import streamlit as st


def indices_for_plot_module(indices: np.ndarray, container: st.container = None) \
        -> (int, bool):
    if container is None:
        container = st.container()

    col1, col2 = container.columns(2)
    n_points = col1.number_input('Number of points to plot', min_value=10, max_value=len(indices),
                                 value=2000 if len(indices) > 2000 else len(indices))
    random = col2.checkbox('Random indices?', value=True)

    return n_points, random


@st.cache
def get_indices(indices: np.ndarray, n_points: int, random: bool):
    if random:
        new_indices = np.random.choice(np.arange(len(indices)), size=n_points)
    else:
        new_indices = np.arange(0, n_points)
    return indices[new_indices]
