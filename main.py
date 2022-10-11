import pandas as pd
import streamlit as st
from omegaconf import OmegaConf

from latent_space_vis.modules.indices_for_plot import indices_for_plot_module, get_indices
from latent_space_vis.modules.jointplot import jointplot_module
from latent_space_vis.modules.subset_selection import subset_selection_module
from latent_space_vis.modules.dim_reduction import dim_reduction_module, get_method, compute_dim_reduction
from latent_space_vis.modules.big_plot import big_plot_module


def main():
    cfg = OmegaConf.load('config.yaml')

    data_df = pd.read_csv(cfg.paths.data, index_col=0)
    label_df = pd.read_csv(cfg.paths.labels, index_col=0)
    for c in label_df.columns:
        label_df[c].apply(str)
    # df = pd.concat((label_df, data_df), axis=1)

    var_df = None
    if cfg.paths.var is not None:
        var_df = pd.read_csv(cfg.paths.var, index_col=0)

    with st.sidebar:
        plot_indices_container = st.container()
        subset_selection_container = st.expander('Subset Selection')

        indices = subset_selection_module(label_df, container=subset_selection_container)

        n_points, random = indices_for_plot_module(indices, plot_indices_container)
        indices = get_indices(indices, n_points, random)

    sub_data_df = data_df.iloc[indices]
    sub_label_df = label_df.iloc[indices]
    sub_df = pd.concat((sub_label_df, sub_data_df), axis=1)

    st.header('Latent space visualization')
    st.subheader('Mean')
    st.table(sub_df.head())

    if var_df is not None:
        st.subheader('Variance')
        st.table(var_df.sample(n=10))

    jointplot_module(sub_df, sub_data_df.columns, label_df)

    st.header('Visualize all dimensions')
    option = st.selectbox('Visualize all dimensions.', options=['Big Plot', 'Dimensionality Reduction'], index=1)

    if option == 'Dimensionality Reduction':
        dim_r_container = st.container()
        method = get_method(dim_r_container)
        reduced_data = compute_dim_reduction(sub_data_df, method)
        dim_reduction_module(reduced_data, sub_label_df, container=dim_r_container)

    elif option == 'Big Plot':
        big_plot_module(sub_df, sub_data_df.columns, sub_label_df)


if __name__ == '__main__':
    main()
