""" Subsections the images in the validation and test sets for oiv_human_verified produced by build_raw_datasets by
    their detected languages
"""

import os

from common import load_csv_as_dict, write_rows_to_csv
from yfcc100m.dataset import detect_language_cld2

raw_dataset_dir = "C:\Honors Project\YFCC100M\dataset"
oiv_human_verified_dir = os.path.join(raw_dataset_dir, "oiv_human_verified")
oiv_human_verified_by_language_dir = os.path.join(raw_dataset_dir, "oiv_human_verified_by_language")
output_dir = os.path.join(raw_dataset_dir, "oiv_human_verified_by_language")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for subset in ["validation", "test"]:
    subset_path = os.path.join(oiv_human_verified_dir, "{}.tsv".format(subset))
    subset_csv = load_csv_as_dict(subset_path, fieldnames=["photo_id", "user_tags", "labels"], delimiter="\t")
    language_rows = {}
    for row in subset_csv:
        image_user_tags = row["user_tags"]
        is_relibable, language = detect_language_cld2(image_user_tags)
        if not is_relibable:
            language = "unknown"
        if not language_rows.get(language):
            language_rows[language] = []
        language_rows[language].append(row)
    for language in language_rows.keys():
        language_subset_path = os.path.join(oiv_human_verified_by_language_dir, "{}-{}.tsv".format(subset, language))
        write_rows_to_csv(language_rows[language], language_subset_path, fieldnames=["photo_id", "user_tags", "labels"],
                          mode="w", delimiter="\t", write_header=False)
