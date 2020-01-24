from yfcc100m.embeddings.prepare import join_dataset_and_autotags, joined_to_subsets
import os

yfcc_dataset_path = "C:\\Honors Project\\YFCC100M\\yfcc100m_dataset"
yfcc_autotags_path = "C:\\Honors Project\\YFCC100M\\yfcc100m_autotags"
output_folder = "C:\\Honors Project\\YFCC100M\\dataset"
raw_dataset_path = os.path.join(output_folder, "raw_dataset")
class_path = "yfcc100m\\embeddings\\relevant_autotags.txt"
oiv_folder = "C:\\Users\\kiera\\OneDrive - University of Edinburgh\\Every Day Files\\Documents\\Microsoft " \
             "Office\\Word\\Homework\\University\\Year 4\\Honours Project\\Open Images\\v5\\Bounding Boxes\\Extended"

print("Build raw dataset")
join_dataset_and_autotags(yfcc_dataset_path, yfcc_autotags_path, raw_dataset_path, class_path=class_path)
print("Build raw subsets")
joined_to_subsets(oiv_folder, raw_dataset_path, output_folder)
