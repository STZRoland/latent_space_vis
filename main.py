import pandas as pd
import streamlit as st
from omegaconf import OmegaConf

from latent_space_vis.modules.indices_for_plot import indices_for_plot_module
from latent_space_vis.modules.jointplot import jointplot_module
from latent_space_vis.modules.subset_selection import subset_selection_module
from latent_space_vis.modules.dim_reduction import dim_reduction_module
from latent_space_vis.modules.big_plot import big_plot_module


def main():
    cfg = OmegaConf.load('config.yaml')

    data_df = pd.read_csv(cfg.paths.data, index_col=0)
    label_df = pd.read_csv(cfg.paths.labels, index_col=0)
    for c in label_df.columns:
        label_df[c].apply(str)
    df = pd.concat((label_df, data_df), axis=1)

    with st.sidebar:
        plot_indices_container = st.container()
        subset_selection_container = st.expander('Subset Selection')

        indices = subset_selection_module(label_df, container=subset_selection_container)
        indices = indices_for_plot_module(indices, plot_indices_container)

    sub_data_df = data_df.iloc[indices]
    sub_label_df = label_df.iloc[indices]
    sub_df = pd.concat((sub_label_df, sub_data_df), axis=1)

    st.header('Latent space visualization')
    st.table(sub_df.head())

    jointplot_module(sub_df, sub_data_df.columns, label_df)

    st.header('Visualize all dimensions')
    option = st.selectbox('Visualize all dimensions.', options=['Big Plot', 'Dimensionality Reduction'], index=1)

    if option == 'Dimensionality Reduction':
        dim_reduction_module(sub_data_df, sub_label_df)

    elif option == 'Big Plot':
        big_plot_module(sub_df, sub_data_df.columns, sub_label_df)


if __name__ == '__main__':
    main()
