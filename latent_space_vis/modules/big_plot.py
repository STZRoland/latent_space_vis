import pandas as pd
import streamlit as st
import seaborn as sns

from latent_space_vis.plots import altair_pairplot


def big_plot_module(df: pd.DataFrame, columns: list[str], label_df: pd.DataFrame = None,
                    container: st.container = None):
    if container is None:
        container = st.container()

    col1, col2 = container.columns(2)

    if label_df is not None:
        hue = col1.selectbox('Hue', options=label_df.columns, index=0, key='hue_3')
    else:
        hue = None

    option = col2.selectbox('Plotting style', options=['Altair', 'Seaborn'], index=0)

    if option == 'Altair':
        chart = altair_pairplot(df, vars=columns, hue=hue)
        container.altair_chart(chart, use_container_width=True)

    elif option == 'Seaborn':
        palette = sns.color_palette()
        g = sns.pairplot(df, hue=hue, vars=columns, palette=palette)
        container.pyplot(g)
