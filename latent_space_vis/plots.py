import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import altair as alt

from latent_space_vis.utils import melt_df

sns.set_style('darkgrid')


def sns_jointplot(df: pd.DataFrame, cols: tuple[str, str], hue: str = None) -> plt.Figure:
    plt.figure()
    palette = sns.color_palette()
    g = sns.jointplot(df, x=cols[0], y=cols[1], hue=hue, palette=palette)
    return g


def altair_jointplot(df: pd.DataFrame, x: str, y: str, hue: str = None, title: str = '', width: int = 500,
                     height: int = 500, scale: tuple[int, int] = None) -> alt.Chart:
    """Creates an altair plot like the seaborn jointplot."""

    colors = alt.Color(
        hue if hue is not None else alt.Undefined,
        type='nominal',
        legend=alt.Legend(
            columns=2,
            # orient='none',
            # legendX=540,
            # legendY=10
        ))

    groupby = [hue] if hue is not None else alt.Undefined
    scale_alt = alt.Scale(domain=alt.Undefined) if scale is None else alt.Scale(domain=scale)

    point_chart = alt.Chart(df).mark_circle().encode(
        x=alt.X(x, scale=scale_alt),
        y=alt.Y(y, scale=scale_alt),
        color=colors,
    ).properties(
        title=title,
        height=height,
        width=width,
    )

    marginal_chart1 = alt.Chart(df).transform_density(
        x,
        as_=[x, 'density'],
        groupby=groupby
    ).mark_line().encode(
        x=alt.X(x + ':Q', axis=None, scale=scale_alt),
        y=alt.Y('density:Q', axis=None),
        color=colors,
    ).properties(
        height=100,
        width=width
    )

    marginal_chart2 = alt.Chart(df).transform_density(
        y,
        as_=[y, 'density'],
        groupby=groupby
    ).mark_line().encode(
        x=alt.X('density:Q', axis=None),
        y=alt.Y(y + ':Q', axis=None, scale=scale_alt),
        order=y,
        color=colors,
    ).properties(
        height=height,
        width=100
    )

    chart = alt.hconcat(point_chart, marginal_chart2)
    chart = alt.vconcat(marginal_chart1, chart)

    return chart


def altair_pairplot(df: pd.DataFrame, vars: list[str], hue: str = None) -> alt.Chart:
    base = alt.Chart().mark_circle().encode(
        color=alt.Color(hue if hue is not None else alt.Undefined, type='nominal')
    )

    chart = alt.concat(data=df)

    for y, y_encoding in enumerate(vars):
        row = alt.hconcat()
        for x, x_encoding in enumerate(vars):
            if x == y:
                row |= altair_kde_plot(df, x_encoding, hue)
            else:
                x_alt = alt.X(x_encoding, axis=alt.Axis(grid=False)) if y == len(vars) - 1 \
                    else alt.X(x_encoding, axis=None)
                y_alt = alt.Y(y_encoding, axis=alt.Axis(grid=False)) if x == 0 \
                    else alt.Y(y_encoding, axis=None)

                row |= base.encode(x=x_alt, y=y_alt)

        chart &= row

    return chart


def altair_kde_plot(df: pd.DataFrame, col: str, hue: str = None, height: int = None, width: int = None) -> alt.Chart:
    chart = alt.Chart(df).transform_density(
        col,
        as_=[col, 'density'],
        groupby=[hue] if hue is not None else alt.Undefined
    ).mark_area(line=True).encode(
        x=alt.X(col + ':Q', axis=None),
        y=alt.Y('density:Q', axis=None),
        color=alt.Color(hue, type='nominal'),
        opacity=alt.value(0.5),
    )

    if height is not None and width is not None:
        chart.properties(
            height=height,
            width=width
        )

    return chart
