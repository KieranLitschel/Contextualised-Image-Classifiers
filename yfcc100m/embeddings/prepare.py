from common import load_csv_as_dict, write_rows_to_csv
from oiv.common import get_train_val_test_flickr_ids
from yfcc100m.common import get_dataset_fields, get_autotag_fields
from yfcc100m.autotags import kept_classes
import re
import os
from tqdm import tqdm


def join_dataset_and_autotags(dataset_path, autotags_path, output_path, keep_numbers=None, class_path=None):
    """ Reads the dataset and autotags files, and writes the id, user tags, and auto tags
        for each image (discarding of videos) to the file at output path, by appending the rows to it

    Parameters
    ----------
    dataset_path : str
        Path to dataset file
    autotags_path : str
        Path to autotags file
    output_path : str
        File to append rows to
    keep_numbers : bool
        Whether to keep numbers, default False
    class_path : str
        Path to classes to keep, to be loaded using kept_classes method. If not specified all classes kept
    """

    keep_numbers = keep_numbers if keep_numbers is not None else False
    dataset = load_csv_as_dict(dataset_path, fieldnames=get_dataset_fields())
    autotags = load_csv_as_dict(autotags_path, fieldnames=get_autotag_fields())
    classes_to_keep = set(kept_classes(class_path)) if class_path else None
    lines = []
    for dataset_row in tqdm(dataset):
        autotags_row = next(autotags)
        image_id = dataset_row["ID"]
        image_user_tags = dataset_row["UserTags"]
        if dataset_row["Video"] == "1":
            continue
        if not keep_numbers:
            image_user_tags = ",".join([tag for tag in image_user_tags.split(",") if not re.match(r"^[0-9]+$", tag)])
        if not image_user_tags:
            continue
        image_auto_tags = autotags_row["PredictedConcepts"]
        if classes_to_keep and image_auto_tags:
            image_auto_tags = ",".join([tag_prob for tag_prob in image_auto_tags.split(",") if
                                        tag_prob.split(":")[0] in classes_to_keep])
        line = "{}\t{}\t{}\n".format(image_id, image_user_tags, image_auto_tags)
        lines.append(line)
        if len(lines) == 10000000:
            output_file = open(output_path, "a")
            output_file.writelines(lines)
            output_file.close()
            lines = []
    if lines:
        lines[-1] = lines[-1][:-1]
        output_file = open(output_path, "a")
        output_file.writelines(lines)
        output_file.close()


def _determine_val_test_ids(path, subset_members):
    """ Determine which flickr id's should belong to the validation and test sets respectively. Such that each is
        a sample of 10% of the number of samples available, and that allocations to train, validation, and test sets
        by OIV are respected

    Parameters
    ----------
    path : str
        Path to file produced by join_dataset_and_autotags
    subset_members : dict of str -> set
        Dict mapping "train", "validation", and "test" to sets, containing the flickr ids of images that belong to them.
        Produced by oiv.common.get_train_val_test_flickr_ids

    Returns
    -------
    (set of str, set of str)
        Set of flickr id's that belong to validation and test respectively
    """

    raw_dataset = load_csv_as_dict(path, fieldnames=["ID", "UserTags", "PredictedConcepts"])
    flickr_ids = []
    for row in tqdm(raw_dataset):
        flickr_ids.append(row["ID"])
    val_test_len = round(len(flickr_ids) * 0.1)
    val_ids = set()
    test_ids = set()
    for flickr_id in flickr_ids:
        if flickr_id in subset_members["validation"]:
            val_ids.add(flickr_id)
        elif flickr_id in subset_members["test"]:
            test_ids.add(flickr_id)
    for flickr_id in flickr_ids:
        if flickr_id in subset_members["train"]:
            continue
        elif len(val_ids) < val_test_len:
            val_ids.add(flickr_id)
        elif len(test_ids) < val_test_len:
            test_ids.add(flickr_id)
        else:
            break
    return val_ids, test_ids


def joined_to_subsets(oiv_folder, dataset_path, output_folder):
    """ Produces train, validation, and test sets for our dataset

    Parameters
    ----------
    oiv_folder : str
        Path to folder containing open images csv's
    dataset_path : str
        Path to file produced by join_dataset_and_autotags
    output_folder : str
        Folder to output the train, validation, and test sets to for the dataset
    """

    print("Getting which images are already assigned to subsets in OIV")
    subset_members = get_train_val_test_flickr_ids(oiv_folder)
    print("Determining samples to be used for validation and test sets")
    val_ids, test_ids = _determine_val_test_ids(dataset_path, subset_members)
    subsets = ["train", "validation", "test"]
    raw_dataset = load_csv_as_dict(dataset_path, fieldnames=["ID", "UserTags", "PredictedConcepts"])
    rows_by_subset = {subset: [] for subset in subsets}
    rows_in_memory = 0
    print("Dividing samples into subsets")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for row in tqdm(raw_dataset):
        flickr_id = row["ID"]
        if flickr_id in val_ids:
            rows_by_subset["validation"].append(row)
        elif flickr_id in test_ids:
            rows_by_subset["test"].append(row)
        elif flickr_id not in subset_members["validation"] and flickr_id not in subset_members["test"]:
            rows_by_subset["train"].append(row)
        rows_in_memory += 1
        if rows_in_memory >= 10000000:
            for subset in subsets:
                write_rows_to_csv(rows_by_subset[subset], os.path.join(output_folder, subset), mode="a")
                rows_by_subset[subset] = []
            rows_in_memory = 0
    for subset in subsets:
        write_rows_to_csv(rows_by_subset[subset], os.path.join(output_folder, subset), mode="a")
