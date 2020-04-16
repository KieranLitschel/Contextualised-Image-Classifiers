""" Computes the mean and standard deviation for the results from the repeats of experiment 1 """

import pathlib
import os
import numpy as np

from common import load_csv_as_dict, write_rows_to_csv

proj_root = pathlib.Path(__file__).parent.parent.absolute()

experiment_1_merged_path = os.path.join(proj_root, "job_results/experiment_1_repeat_merged_summary.csv")
experiment_1_mean_and_std = os.path.join(proj_root, "job_results/experiment_1_with_mean_and_std.csv")

job_name = None
hyper_params = None
mAPs = []

summaries = []

for row in load_csv_as_dict(experiment_1_merged_path, fieldnames=["JobName", "HyperParams", "mAP"], delimiter=","):
    curr_hyper_parms = " ".join(row["HyperParams"].split(" ")[:-1])
    if hyper_params is not None and curr_hyper_parms != hyper_params:
        summaries.append(
            {"JobName": job_name, "HyperParams": hyper_params, "mean mAP": np.mean(mAPs), "std mAP": np.std(mAPs)})
        mAPs = []
    job_name = "_".join(row["JobName"].split("_")[:-1])
    hyper_params = curr_hyper_parms
    mAPs.append(float(row["mAP"]))

write_rows_to_csv(summaries, experiment_1_mean_and_std, fieldnames=["JobName", "HyperParams", "mean mAP", "std mAP"],
                  mode="w", delimiter=",")