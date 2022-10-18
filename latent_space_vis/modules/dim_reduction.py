import numpy as np
import pandas as pd
import streamlit as st

from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE, MDS

from latent_space_vis.plots import altair_jointplot


def dim_reduction_module(reduced_data: np.ndarray, label_df: pd.DataFrame = None, container: st.container = None):
    if container is None:
        container = st.container()

    if label_df is not None:
        hue = container.selectbox('Hue', options=label_df.columns, index=0, key='hue_2')
    else:
        hue = None

    new_df = pd.DataFrame(reduced_data, columns=['dim_1', 'dim_2'], index=label_df.index)
    if label_df is not None:
        new_df = pd.concat([new_df, label_df], axis=1)

    jointplot = altair_jointplot(new_df, x='dim_1', y='dim_2', hue=hue, width=500, height=500)
    container.altair_chart(jointplot, use_container_width=True)


def get_method(container: st.container):
    method = container.selectbox('Select the method (adjust hyperparameters in code).',
                                 options=_options, index=0)
    return method


@st.cache
def compute_dim_reduction(df: pd.DataFrame, method):
    dim_reduction_function = _dim_reduction_dict[method]
    return dim_reduction_function(df)


@st.cache
def pca(data: pd.DataFrame) -> np.ndarray:
    model = PCA(n_components=2)
    return model.fit_transform(data)


@st.cache
def mds(data: pd.DataFrame) -> np.ndarray:
    model = MDS(n_components=2)
    return model.fit_transform(data)


@st.cache
def isomap(data: pd.DataFrame) -> np.ndarray:
    model = Isomap(n_components=2)
    return model.fit_transform(data)


@st.cache
def t_sne(data: pd.DataFrame) -> np.ndarray:
    model = TSNE(n_components=2)
    return model.fit_transform(data)


_dim_reduction_dict = {
    'PCA': pca,
    'Isomap': isomap,
    't-SNE (not distance preserving)': t_sne,
    'MDS (distance preserving)': mds
}
_options = list(_dim_reduction_dict.keys())
