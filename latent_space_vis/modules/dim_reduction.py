import numpy as np
import pandas as pd
import streamlit as st

from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE

from latent_space_vis.plots import altair_jointplot


def dim_reduction_module(df: pd.DataFrame, label_df: pd.DataFrame = None, container: st.container = None):
    if container is None:
        container = st.container()

    col1, col2 = container.columns(2)
    method = col1.selectbox('Select the method.', options=_options, index=0)
    if label_df is not None:
        hue = col2.selectbox('Hue', options=label_df.columns, index=0, key='hue_2')
    else:
        hue = None

    dim_reduction_function = _dim_reduction_dict[method]

    reduced_data = dim_reduction_function(df)
    new_df = pd.DataFrame(reduced_data, columns=['dim_1', 'dim_2'], index=df.index)
    if label_df is not None:
        new_df = pd.concat([new_df, label_df], axis=1)

    jointplot = altair_jointplot(new_df, x='dim_1', y='dim_2', hue=hue, width=500, height=500)
    container.altair_chart(jointplot, use_container_width=True)


@st.cache
def pca(data: pd.DataFrame) -> np.ndarray:
    model = PCA(n_components=2)
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
    't-SNE': t_sne,
}
_options = list(_dim_reduction_dict.keys())
