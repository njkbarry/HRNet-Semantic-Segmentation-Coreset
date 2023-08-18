from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import os
import tensorboard as tb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
from pathlib import Path
from collections import defaultdict


def process_event_acc_path(path: str):
    """
    Extract the name and date of the experiment
    """
    path = Path(path)
    date_rev, name_rev = path.parent.name[::-1].split("_", 1)
    name = name_rev[::-1]
    date = date_rev[::-1]

    return name, date


# Define Scope
SCALARS = ["valid_mIoU"]

# Find paths
top_dir = "/home/nickbarry/Documents/MsC-DS/Data_Science_Research_Project/Coresets/Repositories/HRNet-Semantic-Segmentation-Coreset/log/proxy_experiment/"
tb_paths = []
for root, dirs, files in os.walk(top_dir):
    for file in files:
        if file.endswith(".aa-desktop"):
            tb_paths.append(os.path.join(root, file))

data = defaultdict(list)
run_date_dict = {}

for scalar in SCALARS:
    for tb_path in tb_paths:
        event_acc = EventAccumulator(tb_path)
        event_acc.Reload()
        run_name, date = process_event_acc_path(event_acc.path)
        run_date_dict[run_name] = date
        try:
            events = event_acc.Scalars(scalar)
            for event in events:
                data["w_time"].append(event.wall_time)
                data["step_num"].append(event.step)
                data["val"].append(event.value)
                data["run_name"].append(run_name)
                data["metric"].append(scalar)
        except:
            print(f"tb file {run_name} does not contain results for {scalar}")

    print(0)

df = pd.DataFrame(data)

# Stochastic run plot
run_name = "milo_oracle_spat_sliced_wasserstein_05_a"
run_df = df[df["run_name"] == run_name]

plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
sns.lineplot(data=df, x="step_num", y="val", hue="run_name").set_title("valid_mIoU")

print(0)
