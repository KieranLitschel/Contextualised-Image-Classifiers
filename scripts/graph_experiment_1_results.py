""" Take results from train_embedding_oiv and summarise them in graphs """

import os
import pathlib
import re

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from common import load_csv_as_dict

proj_root = pathlib.Path(__file__).parent.parent.absolute()

experiment_1_merged_path = os.path.join(proj_root, "results/experiment_1_with_mean_and_std.csv")

# y_min is performance of naive baseline, y_max is performance of baseline model on validation set
y_min = 0.66352
y_max = 0.87935

results = list(load_csv_as_dict(experiment_1_merged_path, delimiter=",",
                                fieldnames=["experiment_name", "config", "mean mAP", "std mAP"]))

pad_sizes = set()
tag_threshs = set()
poolings = ["Max", "Average"]

results_by_lang_hyperparams = {}
results_by_pooling = {}
for row in results:
    config = row["config"]
    pad_size = int(re.findall(r"--pad_size=([0-9]+)", config)[0])
    pad_sizes.add(pad_size)
    tag_thresh = int(re.findall(r"--tag_threshold=([0-9]+)", config)[0])
    tag_threshs.add(tag_thresh)
    pooling = re.findall(r"--pooling_layer=([^\s]+)", config)[0]
    if not results_by_lang_hyperparams.get(pad_size):
        results_by_lang_hyperparams[pad_size] = {}
    mean_mAP = float(row["mean mAP"])
    std_mAP = float(row["std mAP"])
    if not results_by_lang_hyperparams[pad_size].get(tag_thresh) or \
            results_by_lang_hyperparams[pad_size][tag_thresh][0] < mean_mAP:
        results_by_lang_hyperparams[pad_size][tag_thresh] = (mean_mAP, std_mAP)
    pooling = "Max" if pooling == "GlobalMaxPool" else "Average"
    if not results_by_pooling.get(pooling) or results_by_pooling[pooling][0] < mean_mAP:
        results_by_pooling[pooling] = (mean_mAP, std_mAP)
results_by_pooling = [(pooling,) + mean_std_mAP for pooling, mean_std_mAP in results_by_pooling.items()]

data = [[pad_size, tag_thresh, mean_std_mAP[0], mean_std_mAP[1]]
        for pad_size, pad_size_dict in results_by_lang_hyperparams.items()
        for tag_thresh, mean_std_mAP in pad_size_dict.items()]

df = pd.DataFrame(data, columns=["Pad Size", "Tag Threshold", "Mean mAP", "Standard Deviation mAP"])
results = df.pivot("Pad Size", "Tag Threshold", "Mean mAP")
sns.heatmap(results, annot=True, fmt=".3g")
plt.savefig("Experiment 1 mean heatmap - Pad Size vs Tag Threshold.pdf")

plt.clf()

results = df.pivot("Pad Size", "Tag Threshold", "Standard Deviation mAP")
sns.heatmap(results, annot=True, fmt=".2g")
plt.savefig("Experiment 1 std heatmap - Pad Size vs Tag Threshold.pdf")

plt.clf()
plt.margins(x=0.4)
plt.xlabel("Pooling Layer")
plt.ylabel("Mean mAP")
df = pd.DataFrame(results_by_pooling,
                  columns=["Pooling Layer", "Mean mAP", "Standard Deviation mAP"])
plt.errorbar(df["Pooling Layer"], df["Mean mAP"], yerr=df["Standard Deviation mAP"], fmt='o')
plt.savefig("Experiment 1 point plot - Pooling Layer.pdf")
