""" Takes the top 3 best results in experiment_1 for each combination of pad_size and tag_threshold, and repeat them
a further two times with different random seeds """

from common import load_csv_as_dict
import re
import pathlib
import os

proj_root = pathlib.Path(__file__).parent.parent.absolute()

results = list(load_csv_as_dict(os.path.join(proj_root, "results/experiment_1_summary.csv"), delimiter=",",
                                fieldnames=["experiment_no", "experiment_name", "config", "map"]))
unique_results = {}
for result in results:
    if not unique_results.get(result["config"]):
        unique_results[result["config"]] = result["map"]

pad_sizes = set()
tag_threshs = set()

output_folder = "/home/s1614973/HonorsProject/Embeddings/experiment_1"

results_by_lang_hyperparams = {}
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

job_script = open(
    os.path.join(proj_root, "milano_configs/embeddings_experiment_run.sh"),
    "r").read()

for escaped_symbol in ["$", '"', "`"]:
    job_script = re.sub(r"\\\{}".format(escaped_symbol), escaped_symbol, job_script)

for pad_size in pad_sizes:
    for tag_thresh in tag_threshs:
        top_three = sorted(list(results_by_lang_hyperparams[pad_size][tag_thresh].items()), key=lambda x: x[1],
                           reverse=True)[:3]
        job_no = 0
        for config, _ in top_three:
            for seed in range(1, 3):
                job_name = "repeat_{}_{}_{}_{}".format(pad_size, tag_thresh, job_no, seed)
                job_output_folder = os.path.join(output_folder, job_name)
                if os.path.exists(job_output_folder):
                    if os.path.exists(os.path.join(job_output_folder, "best_model_validation_metrics.csv")):
                        continue
                    else:
                        os.system("rm -rf {}".format(job_output_folder))
                new_script = job_script.replace("--random_seed 0", "--random_seed {}".format(seed)) \
                    .replace('"$@"', config).replace("${SLURM_JOB_NAME%???}", job_name)
                with open("{}.sh".format(job_name), "w") as f:
                    f.write(new_script)
                os.system("sbatch {}.sh".format(job_name))
            job_no += 1
