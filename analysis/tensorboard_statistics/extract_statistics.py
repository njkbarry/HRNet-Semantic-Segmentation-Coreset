from packaging import version
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
import os
import tensorboard as tb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
from pathlib import Path
from collections import defaultdict
from tensorboard.util.tensor_util import make_ndarray
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing


def process_event_acc_path(path: str):
    """
    Extract the name and date of the experiment
    """
    path = Path(path)
    date_rev, name_rev = path.parent.name[::-1].split("_", 1)
    name = name_rev[::-1]
    date = date_rev[::-1]

    coreset_algorithm = None
    repitition = None
    coreset_frac = None
    metric = None
    embedding = None

    if name[-1] in ["a", "b", "c"]:
        coreset_algorithm = name.split("_")[0]
        repitition = name.split("_")[-1]
        coreset_frac = name.split("_")[-2]
        if "sliced_wasserstein" in name:
            metric = "sliced_wasserstein"
        elif "cossim" in name:
            metric = "cossim"
        elif "gromov_wasserstein" in name:
            metric = "gromov_wasserstein"
        elif "fronerbius" in name:
            metric = "fronerbius"
        else:
            metric = "cossim"

        if "oracle_spat" in name:
            embedding = "oracle_spat"
        elif "oracle_context" in name:
            embedding = "oracle_context"
        elif "segformer" in name:
            embedding = "segformer"
        elif "ViT" in name:
            embedding = "ViT"
        else:
            embedding = "unknown"

    return name, date, coreset_algorithm, repitition, coreset_frac, metric, embedding


# Define Scope
SCALARS = ["valid_mIoU", "valid_loss", "train_loss"]  # + [f"valid_mIoU_class_{i}" for i in range(60)]
# SCALARS = [f"valid_mIoU_class_{i}" for i in range(60)]

# Find paths
top_dir = "/home/nickbarry/Documents/MsC-DS/Data_Science_Research_Project/Coresets/Repositories/HRNet-Semantic-Segmentation-Coreset/log/"
tb_paths = []
for root, dirs, files in os.walk(top_dir):
    for file in files:
        if file.endswith(".aa-desktop"):
            tb_paths.append(os.path.join(root, file))

data = defaultdict(list)
run_date_dict = {}
full_model_performance_dict = dict(zip(SCALARS, [0.49, 0.8753, 0.1515]))

for scalar in SCALARS:
    data = defaultdict(list)
    for tb_path in tb_paths:
        event_acc = EventAccumulator(tb_path)
        event_acc.Reload()
        (
            run_name,
            date,
            coreset_algorithm,
            repitition,
            coreset_frac,
            sim_metric,
            embedding,
        ) = process_event_acc_path(event_acc.path)
        run_date_dict[run_name] = date
        experiment_name = None
        epsilon = None
        base_set_threshold = None
        run_name = Path(tb_path).parts[-2].rsplit("_", 1)[0].rsplit("_", 1)[0]
        if len(event_acc.Tags()["tensors"]) > 0:
            # Parse hyperparameters from tb
            try:
                coreset_algorithm = make_ndarray(event_acc.Tensors("CORESET_ALGORITHM/text_summary")[0].tensor_proto)[0].decode()
                coreset_frac = make_ndarray(event_acc.Tensors("RANDOM_SUBSET/text_summary")[0].tensor_proto)[0].decode().strip(".")
                sim_metric = make_ndarray(event_acc.Tensors("METRIC/text_summary")[0].tensor_proto)[0].decode()
                embedding = make_ndarray(event_acc.Tensors("FEATURE_EMBEDDER/text_summary")[0].tensor_proto)[0].decode()
                experiment_name = make_ndarray(event_acc.Tensors("EXPERIMENT_NAME/text_summary")[0].tensor_proto)[0].decode()
                epsilon = make_ndarray(event_acc.Tensors("EPSILON/text_summary")[0].tensor_proto)[0].decode()
                base_set_threshold = make_ndarray(event_acc.Tensors("BASE_SET_THRESHOLD/text_summary")[0].tensor_proto)[0].decode()

            except Exception as e:
                pass
        try:
            events = event_acc.Scalars(scalar)
            for event in events:
                data["w_time"].append(event.wall_time)
                data["step_num"].append(event.step)
                data["val"].append(event.value)
                data["run_name"].append(run_name)
                data["metric"].append(scalar)
                data["coreset_algorithm"].append(coreset_algorithm)
                data["repitition"].append(repitition)
                data["coreset_frac"].append(coreset_frac)
                data["sim_metric"].append(sim_metric)
                data["embedding"].append(embedding)
                data["experiment_name"].append(experiment_name)
                data["epsilon"].append(epsilon)

        except KeyError:
            print(f"tb file {run_name} does not contain results for {scalar}")

    print(0)

    df = pd.DataFrame(data)
    # Stochastic run plot
    # plot_df = df[(df["coreset_algorithm"] == "adaptiverandom") | (df["coreset_algorithm"] == "craig")]
    plot_df = df[df["experiment_name"].isin(["pixel_map_weighted_random_sampling_experiment", "adaptive_random_profiling"])]
    # plot_df = df[df["repitition"].notnull()]
    # plot_df = df[df["experiment_name"].isin(["gradient_approximation"])]
    # plot_df = df[df["run_name"].isin(["craig_ViT_05", "static_random_05"])]
    # plot_df = plot_df[plot_df["coreset_algorithm"].isin(["craig", "adaptiverandom"])]
    plot_df = plot_df[plot_df["coreset_frac"].str.endswith("5")]
    # plot_df["step_num"].replace(0, 1, inplace=True)
    plot_df.loc[:, "step_num"] = plot_df["step_num"] + 1
    plot_df.loc[:, "step_num"] = plot_df["step_num"] * 3

    hue = plot_df["experiment_name"].astype(str) + ", " + plot_df["run_name"].astype(str) + ", " + plot_df["coreset_algorithm"].astype(str)
    sns.set_style("darkgrid")
    smoothing = False
    if smoothing:
        pass
        plot_df["val"] = Holt(plot_df["val"]).fit(smoothing_level=0.9, smoothing_slope=0.5, optimized=True)._fittedvalues
        plot_df["val"] = ExponentialSmoothing(plot_df["val"]).fit(smoothing_level=0.9, smoothing_slope=0.9)._fittedvalues

    line_plt = sns.lineplot(data=plot_df, x="step_num", y="val", hue=hue)  # .set_title("valid_mIoU")
    line_plt.set_ylim(np.min(line_plt.get_yticks()) * 0.7, np.max(line_plt.get_yticks()) * 1.15)
    line_plt.get_yaxis().set_minor_locator(ticker.AutoMinorLocator(n=10))
    line_plt.grid(which="major", color="w", linewidth=1.0)
    line_plt.grid(which="minor", color="w", linewidth=0.5)
    line_plt.axhline(full_model_performance_dict[scalar], alpha=0.5, color="red")
    line_plt.axhline(full_model_performance_dict[scalar] * (0.95 if scalar == "valid_mIoU" else 1.05), alpha=0.5, color="red", linestyle="--")
    line_plt.axhline(full_model_performance_dict[scalar] * (0.90 if scalar == "valid_mIoU" else 1.10), alpha=0.5, color="red", linestyle=":")

    line_plt.set_xlabel("Epoch")
    line_plt.set_ylabel(scalar)
    line_plt.set_title("stochastic_sampling_epsilon_experiment")

    plt.show()
    plt.close()

print(0)
