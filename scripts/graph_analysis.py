import math
import os
import pathlib

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker

from oiv.common import get_hierarchy_json_path, get_class_descriptions_path, get_hierarchy_by_level

proj_root = pathlib.Path(__file__).parent.parent.absolute()

experiment_maps = {"Original": [0.7481407242026616, 0.748721526, 0.749476923],
                   "Fine-Tuned": [0.733641353296024, 0.730644964750732, 0.729181126395337],
                   "Most Likely Class": [0.635850179],
                   "Original Floored": [0.647748983982311, 0.645886573764493, 0.646263863153314],
                   "Machine-Generated Labels": [0.878656405867957],
                   "English Best Model": [0.746953834, 0.743078918, 0.745654112],
                   "English Fine-Tuned": [0.728503208, 0.729714927, 0.72694073],
                   "Non-English Best Model": [0.77260902, 0.771070024, 0.773496219],
                   "Non-English Fine-Tuned": [0.763937961, 0.760464632, 0.764313694]}

graphs = [("Original", "Most Likely Class"), ("Original Floored", "Machine-Generated Labels"),
          ("English Best Model", "English Fine-Tuned"), ("Non-English Best Model", "Non-English Fine-Tuned")]

for pair in graphs:
    results = [[name, np.mean(maps), np.std(maps)] for name, maps in
               {name: experiment_maps[name] for name in pair}.items()]
    df = pd.DataFrame(results, columns=["Experiment", "Mean mAP", "Standard Deviation mAP"])
    plt.margins(x=0.4)
    plt.xlabel("Experiment")
    plt.ylabel("Mean mAP")
    plt.errorbar(df["Experiment"], df["Mean mAP"], yerr=df["Standard Deviation mAP"], fmt='.')
    plt.savefig("Experiment Analysis point plot - {} vs {}.pdf".format(pair[0], pair[1]))
    plt.clf()

df = pd.read_csv(os.path.join(proj_root, "baselines/most freq class baseline/test.csv"))
machine_class_aps = {row[1][1].split("/")[-1]: (float(row[1][2]), 0) for row in list(df.iterrows())[1:] if
                     not math.isnan(float(row[1][2]))}

best_model_class_aps = {}
for i in range(3):
    df = pd.read_csv(os.path.join(proj_root, "experiment_results/basic/repeat {}/"
                                             "best_model_floored_false_preds_test_metrics.csv".format(i)))
    best_model_class_aps[i] = {row[1][1].split("/")[-1]: float(row[1][2]) for row in list(df.iterrows())[1:] if
                               not math.isnan(float(row[1][2]))}

best_model_class_mean_aps = {label: (np.mean([best_model_class_aps[i][label] for i in range(3)]),
                                     np.std([best_model_class_aps[i][label] for i in range(3)]))
                             for label in best_model_class_aps[0].keys()}

hierarchy_by_level = get_hierarchy_by_level(get_hierarchy_json_path(), get_class_descriptions_path())

for name, aps in [("Machine Generated", machine_class_aps), ("Best Model", best_model_class_mean_aps)]:
    results = [[level] + list(aps[label]) for level, labels in hierarchy_by_level.items() for label in labels if
               aps.get(label) is not None]
    df = pd.DataFrame(results, columns=["Level", "Mean mAP", "Standard Deviation mAP"])
    plt.xlabel("Level")
    plt.ylabel("Mean AP")
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.errorbar(df["Level"], df["Mean mAP"], yerr=df["Standard Deviation mAP"], fmt='.')
    plt.savefig("Experiment Analysis point plot - {} mean AP by level.pdf".format(name))
    plt.clf()
