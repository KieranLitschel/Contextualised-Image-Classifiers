# Compute the AP for each class excluding where machine-generated labels and predictions agree, we did not include
# these results in our report as they turned out not to be useful

import tensorflow as tf
import pathlib
import os
from scripts.train_embedding_oiv import model_preds, evaluate_model
from oiv.evaluate import machine_labels_baseline, build_y_pred_machine_labels
from embeddings.encoders import CommaTokenTextEncoder
import pandas as pd
from tqdm import tqdm

proj_root = pathlib.Path(__file__).parent.parent.absolute()
models_dir = os.path.join(proj_root, "experiment_results/basic")
machine_generated_labels_baseline_path = os.path.join(proj_root, "baselines/machine generated labels baseline")
oiv_dataset_dir = "C:\\Honors Project\\YFCC100M\\dataset\\oiv"
oiv_human_dataset_dir = "C:\\Honors Project\\YFCC100M\\dataset\\oiv_human_verified"
oiv_csv_dir = "C:\\Users\\kiera\\OneDrive - University of Edinburgh\\Every Day Files\\Documents\\Microsoft " \
              "Office\\Word\\Homework\\University\\Year 4\\Honours Project\\Open Images\\v5\\Mixture"

classes_encoder_path = os.path.join(proj_root, "experiment_results/classes_encoder")
classes_encoder = CommaTokenTextEncoder.load_from_file(classes_encoder_path)
pad_size = 20

for i in tqdm(range(3)):
    model_dir = os.path.join(models_dir, "repeat {}".format(i))
    features_encoder = CommaTokenTextEncoder.load_from_file(os.path.join(model_dir, "features_encoder"))
    with tf.Session():
        y_pred_model = model_preds(model_dir, oiv_human_dataset_dir, pad_size, features_encoder, classes_encoder,
                                   floor_false_preds=True, subset="test")
    y_pred_machine = build_y_pred_machine_labels("test", oiv_csv_dir, oiv_human_dataset_dir, classes_encoder)
    machine_metrics = machine_labels_baseline("test", oiv_csv_dir, oiv_human_dataset_dir,
                                              os.path.join(proj_root, "oiv/class-descriptions-boxable.csv"),
                                              classes_encoder,
                                              y_pred_benchmark=y_pred_model)
    df_metrics = pd.DataFrame(list(machine_metrics.items()))
    df_metrics.to_csv(os.path.join(machine_generated_labels_baseline_path,
                                   "best_model_{}_agreement_removed_test_metrics.csv".format(i)))
    evaluate_model(model_dir, classes_encoder_path, oiv_human_dataset_dir, pad_size, floor_false_preds=True,
                   output_file_name="best_model_floored_false_preds_agreement_removed_test_metrics.csv", subset="test",
                   y_pred_benchmark=y_pred_machine)
