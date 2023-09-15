import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def make_line_plot(figure_df, x_column, y_column, hue, x_label, y_label, title):
    line_plt = sns.lineplot(data=figure_df, x=x_column, y=y_column, hue=hue)  # .set_title("valid_mIoU")
    line_plt.set_ylim(np.min(line_plt.get_yticks()) * 0.7, np.max(line_plt.get_yticks()) * 1.15)
    line_plt.set_xlabel(x_label)
    line_plt.set_ylabel(y_label)
    line_plt.set_title(title)
