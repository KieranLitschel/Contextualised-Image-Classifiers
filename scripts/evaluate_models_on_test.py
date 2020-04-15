from scripts.train_embedding_oiv import evaluate_model
import os
from tqdm import tqdm

pad_size = 20
dataset_root_dir = "/home/s1614973/HonorsProject/Embeddings/dataset"
classes_encoder_path = os.path.join(dataset_root_dir, "classes_encoder")
oiv_dataset_dir = os.path.join(dataset_root_dir, "oiv")
oiv_human_dataset_dir = os.path.join(dataset_root_dir, "oiv_human_verified")
best_models_dir = "/home/s1614973/HonorsProject/Embeddings/best_models"

for model_dir, _, files in tqdm(os.walk(best_models_dir)):
    if "best_model.h5" in files:
        if "best_model_validation_metrics.csv" not in files:
            evaluate_model(model_dir, classes_encoder_path, oiv_human_dataset_dir, pad_size)
        if "best_model_test_metrics.csv" not in files:
            evaluate_model(model_dir, classes_encoder_path, oiv_human_dataset_dir, pad_size,
                           output_file_name="best_model_test_metrics.csv", subset="test")
        if "best_model_floored_false_preds_test_metrics.csv" not in files:
            evaluate_model(model_dir, classes_encoder_path, oiv_human_dataset_dir, pad_size,
                           output_file_name="best_model_floored_false_preds_test_metrics.csv", subset="test",
                           floor_false_preds=True)
        if "best_model_test_english_metrics.csv" not in files:
            evaluate_model(model_dir, classes_encoder_path, oiv_human_dataset_dir+"_english", pad_size,
                           output_file_name="best_model_test_english_metrics.csv", subset="test")
        if "best_model_test_other_metrics.csv" not in files:
            evaluate_model(model_dir, classes_encoder_path, oiv_human_dataset_dir+"_other", pad_size,
                           output_file_name="best_model_test_other_metrics.csv", subset="test")
