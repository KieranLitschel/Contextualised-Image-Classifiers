""" Take results from train_embedding_oiv and summarise them in graphs """

from common import load_csv_as_dict
import re
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# y_min is performance of naive baseline, y_max is performance of baseline model on validation set
y_min = 0.66352
y_max = 0.87935

results = list(load_csv_as_dict("C:\\Honors Project\\Results\\Experiment 1\\results.csv", delimiter=",",
                                fieldnames=["experiment_no", "experiment_name", "config", "map"]))
unique_results = {}
for result in results:
    if not unique_results.get(result["config"]):
        unique_results[result["config"]] = result["map"]

pad_sizes = set()
tag_threshs = set()
poolings = ["Max", "Average"]

results_by_lang_hyperparams = {}
results_by_pooling = []
for config, mAP in unique_results.items():
    pad_size = int(re.findall(r"--pad_size=([0-9]+)", config)[0])
    pad_sizes.add(pad_size)
    tag_thresh = int(re.findall(r"--tag_threshold=([0-9]+)", config)[0])
    tag_threshs.add(tag_thresh)
    pooling = re.findall(r"--pooling_layer=([^\s]+)", config)[0]
    if not results_by_lang_hyperparams.get(pad_size):
        results_by_lang_hyperparams[pad_size] = {}
    if not results_by_lang_hyperparams[pad_size].get(tag_thresh):
        results_by_lang_hyperparams[pad_size][tag_thresh] = {}
    mAP = float(mAP)
    results_by_lang_hyperparams[pad_size][tag_thresh][config] = mAP
    results_by_pooling.append(("Max" if pooling == "GlobalMaxPool" else "Average", pad_size, tag_thresh, mAP))

data = [[pad_size, tag_thresh, len(tag_thresh_dict), max(tag_thresh_dict.values())]
        for pad_size, pad_size_dict in results_by_lang_hyperparams.items()
        for tag_thresh, tag_thresh_dict in pad_size_dict.items()]

df = pd.DataFrame(data, columns=["Pad Size", "Tag Threshold", "number of results", "Best mAP"])
results = df.pivot("Pad Size", "Tag Threshold", "Best mAP")
sns.heatmap(results, annot=True, vmin=y_min, vmax=y_max)
plt.savefig("Experiment 1 heatmap - Pad Size vs Tag Threshold.pdf")

plt.clf()
df = pd.DataFrame(results_by_pooling, columns=["Pooling Layer", "Pad Size", "Tag Threshold", "mAP"])
sns.pointplot(x="Pooling Layer", y="mAP", data=df, join=False)
plt.savefig("Experiment 1 point plot - Pooling Layer.pdf")

for other_param, values in [("Pad Size", pad_sizes), ("Tag Threshold", tag_threshs)]:
    plt.clf()
    pad_size_df = pd.DataFrame(
        [(param_value, pooling, df[(df["Pooling Layer"] == pooling) & (df[other_param] == param_value)]["mAP"].max())
         for pooling in poolings for param_value in values], columns=[other_param, "Pooling Layer", "Best mAP"])
    results = pad_size_df.pivot("Pooling Layer", other_param, "Best mAP")
    sns.heatmap(results, annot=True, vmin=y_min, vmax=y_max)
    plt.savefig("Experiment 1 heat map - Pooling Layer vs {}.pdf".format(other_param))
