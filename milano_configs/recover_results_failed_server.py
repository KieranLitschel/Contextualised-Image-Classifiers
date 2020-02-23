# used when Milano server dies, to gather together the result from jobs that didn't fail

import argparse
import os
import re
from common import write_rows_to_csv

parser = argparse.ArgumentParser()

parser.add_argument("--experiment_name",
                    help="Name of experiment that was being run", type=str)
parser.add_argument("--new_experiment_name", help="Name of experiment to move results to")
parser.add_argument("--clean_up", help="Set this flag if this script has already been run, and just need to gather"
                                       "results for jobs that were incomplete when first run", dest="clean_up",
                    action="store_true")

script_args = parser.parse_args()

root = "/home/s1614973"
experiment_root = "/home/s1614973/HonorsProject/Embeddings"

old_experiment = os.path.join(experiment_root, script_args.experiment_name)
new_experiment = os.path.join(experiment_root, script_args.new_experiment_name)
results_csv = os.path.join(new_experiment, "results.csv")

if not os.path.exists(new_experiment):
    os.makedirs(new_experiment)

if not script_args.clean_up:
    files = "\n".join(os.listdir(root))
    script_files = re.findall(r"(script-[0-9]+\.[0-9]+-([0-9]+))\.sh", files)
else:
    files = "\n".join(os.listdir(new_experiment))
    script_files = re.findall(r"(script-[0-9]+\.[0-9]+-([0-9]+))\.sh", files)

for job_name, job_id in script_files:
    job_script_path = os.path.join(root if not script_args.clean_up else new_experiment, "{}.sh".format(job_name))
    job_output_folder = os.path.join(old_experiment, job_name)
    job_results_file = os.path.join(job_output_folder, "best_model_validation_metrics.csv")
    if not os.path.exists(job_results_file):
        os.system("mv {} {}".format(job_script_path, new_experiment))
    else:
        mean_average_precision = \
            re.findall(r"mAP@0\.5IOU,([0-9]+\.[0-9]+)", open(job_results_file, "r").readlines()[1])[0]
        hyper_params = \
            re.findall(r"--pad_size=[0-9]+ --tag_threshold=[0-9]+ --layer_capacity=[0-9]+ "
                       r"--pooling_layer=(?:(?:GlobalAveragePooling)|(?:GlobalMaxPool)) "
                       r"--batch_size=[0-9]+ --learning_rate=[0-9]+\.[0-9]+ --dropout_rate=[0-9]+\.[0-9]+ "
                       r"--l2_reg_factor=[0-9]+\.[0-9]+",
                       "\n".join(open(job_script_path, "r").readlines()))[0]
        result = {"JobID": job_id, "JobName": job_name, "HyperParams": hyper_params, "mAP": mean_average_precision}
        write_rows_to_csv([result], results_csv, mode="a", delimiter=",",
                          fieldnames=["JobID", "JobName", "HyperParams", "mAP"])
        os.system("mv {} {}".format(job_output_folder, new_experiment))
        new_job_output_folder = os.path.join(new_experiment, job_name)
        os.system("mv {} {}".format(job_script_path, new_job_output_folder))
