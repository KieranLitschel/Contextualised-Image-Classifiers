# used to compile results of repeated experiments

import os
import re
from common import write_rows_to_csv
from tqdm import tqdm
import pickle

experiment_path = "/home/s1614973/HonorsProject/Embeddings/experiment_1_repeat"

results_csv = os.path.join(experiment_path, "results.csv")

folders = os.listdir(experiment_path)

for folder_name in tqdm(folders):
    job_output_folder = os.path.join(experiment_path, folder_name)
    job_results_file = os.path.join(job_output_folder, "best_model_validation_metrics.csv")
    with open(os.path.join(job_output_folder, "script_args.pickle"), "rb") as script_args_file:
        script_args = pickle.load(script_args_file)
    mean_average_precision = \
        re.findall(r"mAP@0\.5IOU,([0-9]+\.[0-9]+)", open(job_results_file, "r").readlines()[1])[0]
    hyper_params = "--pad_size={} --tag_threshold={} --layer_capacity={} --pooling_layer={} --batch_size={} " \
                   "--learning_rate={} --dropout_rate={} --random_seed={}".format(script_args.pad_size,
                                                                                  script_args.tag_threshold,
                                                                                  script_args.layer_capacity,
                                                                                  script_args.pooling_layer,
                                                                                  script_args.batch_size,
                                                                                  script_args.learning_rate,
                                                                                  script_args.dropout_rate,
                                                                                  script_args.random_seed)
    result = {"JobName": folder_name, "HyperParams": hyper_params, "mAP": mean_average_precision}
    write_rows_to_csv([result], results_csv, mode="a", delimiter=",",
                      fieldnames=["JobName", "HyperParams", "mAP"])
