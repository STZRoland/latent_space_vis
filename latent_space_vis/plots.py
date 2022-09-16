import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style('darkgrid')


def sns_jointplot(df: pd.DataFrame, cols: tuple[str, str], hue: [str] = None) -> plt.Figure:
    plt.figure()
    palette = sns.color_palette()
    g = sns.jointplot(df, x=cols[0], y=cols[1], hue=hue, palette=palette)
    return g
