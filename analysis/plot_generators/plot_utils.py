import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import pickle

SCALARS = ["valid_mIoU", "valid_loss", "train_loss"]
PASCAL_PERFORMANCE_DICT = dict(zip(SCALARS, [0.49, 0.8753, 0.1515]))


def make_line_plot(figure_df, x_column, y_column, hue, x_label, y_label, title):
    line_plt = sns.lineplot(data=figure_df, x=x_column, y=y_column, hue=hue)  # .set_title("valid_mIoU")
    line_plt.set_ylim(np.min(line_plt.get_yticks()) * 0.7, np.max(line_plt.get_yticks()) * 1.15)
    line_plt.set_xlabel(x_label)
    line_plt.set_ylabel(y_label)
    line_plt.set_title(title)


def make_performance_line_plot(figure_df, x_column, scalar, hue, title):
    assert scalar in SCALARS, f"{scalar} not defined for performance plotting"
    line_plt = sns.lineplot(data=figure_df, x=x_column, y=scalar, hue=hue, palette="colorblind")
    line_plt.set_ylim(np.min(line_plt.get_yticks()) * 0.7, np.max(line_plt.get_yticks()) * 1.15)
    line_plt.get_yaxis().set_minor_locator(ticker.AutoMinorLocator(n=10))
    line_plt.grid(which="major", color="w", linewidth=1.0)
    line_plt.grid(which="minor", color="w", linewidth=0.5)
    line_plt.axhline(PASCAL_PERFORMANCE_DICT[scalar], alpha=0.5, color="red")
    line_plt.axhline(PASCAL_PERFORMANCE_DICT[scalar] * (0.95 if scalar == "valid_mIoU" else 1.05), alpha=0.5, color="red", linestyle="--")
    line_plt.axhline(PASCAL_PERFORMANCE_DICT[scalar] * (0.90 if scalar == "valid_mIoU" else 1.10), alpha=0.5, color="red", linestyle=":")

    line_plt.set_xlabel("Epoch")
    line_plt.set_ylabel(scalar)
    line_plt.set_title(title)


def jaccard_set(list1, list2):
    """Define Jaccard Similarity function for two sets"""
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


def pickle2dict(file_name, key):
    """
    Load dictionary from pickle file
    """
    with open(file_name, "rb") as fIn:
        stored_data = pickle.load(fIn)
        value = stored_data[key]
    return value
