""" Fetches the mAP from original experiments that used seed 0 to combine with repeated experiments for seed 1 and 2 """

import pathlib
import os

from common import load_csv_as_dict, write_rows_to_csv

proj_root = pathlib.Path(__file__).parent.parent.absolute()

experiment_1_path = os.path.join(proj_root, "job_results/experiment_1_summary.csv")
experiment_1_repeat_path = os.path.join(proj_root, "job_results/experiment_1_repeat_summary.csv")
experiment_1_repeat_merged_path = os.path.join(proj_root, "job_results/experiment_1_repeat_merged_summary.csv")

experiment_1_repeat_results = sorted([row for row in load_csv_as_dict(experiment_1_repeat_path,
                                                                      fieldnames=["JobName", "HyperParams", "mAP"],
                                                                      delimiter=",")], key=lambda x: x["JobName"])
experiment_1_results = {" ".join(row["HyperParams"].split(" ")[:-1]): row for row in
                        load_csv_as_dict(experiment_1_path,
                                         fieldnames=["JobNumber", "ScriptName", "HyperParams", "mAP"],
                                         delimiter=",")}

combined_results = []
seen_hyper_params = set()
for row in experiment_1_repeat_results:
    if row["HyperParams"] in seen_hyper_params:
        continue
    if row["JobName"].split("_")[-1] == "1":
        hyper_parms = " ".join(row["HyperParams"].split(" ")[:-1])
        seed_0_row = experiment_1_results[hyper_parms]
        del seed_0_row["JobNumber"]
        del seed_0_row["ScriptName"]
        seed_0_row["JobName"] = "_".join(row["JobName"].split("_")[:-1] + ["0"])
        seed_0_row["HyperParams"] = " ".join(row["HyperParams"].split(" ")[:-1] + ["--random_seed=0"])
        combined_results.append(seed_0_row)
    combined_results.append(row)
    seen_hyper_params.add(row["HyperParams"])

write_rows_to_csv(combined_results, experiment_1_repeat_merged_path, fieldnames=["JobName", "HyperParams", "mAP"],
                  mode="w", delimiter=",")
