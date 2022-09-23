import pandas as pd
import streamlit as st

from latent_space_vis.plots import altair_jointplot


def jointplot_module(df: pd.DataFrame, dimensions: list[str], label_df: pd.DataFrame = None,
                     container: st.container = None):
    if container is None:
        container = st.container()
    container.header('Plot the individual dimensions.')

    col1, col2, col3, col4 = container.columns(4)
    dim_1 = col1.selectbox('X-axis', options=dimensions, index=0)
    dim_2 = col2.selectbox('Y-axis', options=dimensions, index=1)

    limit_axes = col4.checkbox('Set scale to [-1, 1]?', value=False)
    scale = (-1, 1) if limit_axes else None

    if label_df is not None:
        hue = col3.selectbox('Hue', options=label_df.columns, index=0, key='hue_1')
    else:
        hue = None

    jointplot = altair_jointplot(df, x=dim_1, y=dim_2, hue=hue, width=500, height=500, scale=scale)
    container.altair_chart(jointplot, use_container_width=True)
