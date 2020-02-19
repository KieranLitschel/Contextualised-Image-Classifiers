import os
from embeddings.prepare import count_classes_in_subset
from common import write_rows_to_csv
from oiv.common import hierarchy_members_list, get_hierarchy_json_path

dataset_dir = "C:\\Honors Project\\YFCC100M\\dataset"
individual_class_counts = {}

for (dirpath, dirnames, filenames) in os.walk(dataset_dir):
    if filenames:
        dir_name = dirpath.split("\\")[-1]
        individual_class_counts[dir_name] = {}
        for filename in filenames:
            subset_name, ext = filename.split(".")
            if ext == "tsv" and subset_name in ["train", "validation", "test"]:
                individual_class_counts[dir_name][subset_name] = count_classes_in_subset(
                    os.path.join(dirpath, filename))

classes = hierarchy_members_list(get_hierarchy_json_path())
fieldnames = ["Label"] + ["{}\\{}".format(dir_name, subset) for dir_name in individual_class_counts.keys() for subset in
                          individual_class_counts[dir_name].keys()]
class_counts = []
for label in classes:
    class_count = {"Label": label}
    for dir_name in individual_class_counts.keys():
        for subset in individual_class_counts[dir_name].keys():
            class_count["{}\\{}".format(dir_name, subset)] = individual_class_counts[dir_name][subset].get(label, 0)
    class_counts.append(class_count)

write_rows_to_csv(class_counts, os.path.join(dataset_dir, "class_counts.csv"), fieldnames=fieldnames, write_header=True,
                  delimiter=",")
