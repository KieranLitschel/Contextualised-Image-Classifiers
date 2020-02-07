from embeddings.prepare import join_dataset_and_autotags, joined_to_subsets
import os

yfcc_dataset_path = "C:\\Honors Project\\YFCC100M\\yfcc100m_dataset"
yfcc_autotags_path = "C:\\Honors Project\\YFCC100M\\yfcc100m_autotags"
output_folder = "C:\\Honors Project\\YFCC100M\\dataset"
oiv_folder = "C:\\Users\\kiera\\OneDrive - University of Edinburgh\\Every Day Files\\Documents\\Microsoft " \
             "Office\\Word\\Homework\\University\\Year 4\\Honours Project\\Open Images\\v5\\Mixture"
raw_dataset_yfcc_output = os.path.join(output_folder, "raw_dataset_yfcc")
raw_dataset_oiv_human_machine_output = os.path.join(output_folder, "raw_dataset_oiv_human_machine")
raw_dataset_oiv_human_output = os.path.join(output_folder, "raw_dataset_oiv_human")
stem = False

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("Build raw dataset")
join_dataset_and_autotags(yfcc_dataset_path, yfcc_autotags_path, oiv_folder, raw_dataset_yfcc_output,
                          raw_dataset_oiv_human_machine_output, raw_dataset_oiv_human_output, stem=stem)
