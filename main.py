import numpy as np
import pandas as pd
import streamlit as st
from omegaconf import OmegaConf

from latent_space_vis.utils import get_float_columns
from latent_space_vis.plots import sns_jointplot


def main():
    cfg = OmegaConf.load('config.yaml')

    df = pd.read_csv(cfg.paths.data)
    if cfg.latent_columns is None:
        latent_columns = get_float_columns(df)
    else:
        latent_columns = cfg.latent_columns

    st.header('Latent space visualization')
    st.table(df.head())

    with st.sidebar:
        col1, col2 = st.columns(2)
        n_points = col1.number_input('Number of points to plot', min_value=10, max_value=len(df), value=1000)
        random = col2.checkbox('Random indices?', value=True)
        if random:
            indices = np.random.choice(np.arange(len(df)), size=n_points)
        else:
            indices = np.arange(0, n_points)

    plot_df = df.iloc[indices]

    col1, col2 = st.columns(2)
    dim_1 = col1.selectbox('X-axis', options=latent_columns, index=0)
    dim_2 = col2.selectbox('Y-axis', options=latent_columns, index=1)

    st.pyplot(sns_jointplot(plot_df, (dim_1, dim_2), hue='athlete_id'))


if __name__ == '__main__':
    main()
