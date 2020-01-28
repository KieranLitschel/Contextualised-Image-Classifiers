import re
import os
from common import load_csv_as_dict
from tqdm import tqdm
import json


def extract_image_id_from_flickr_static(static_url):
    """ Given a static url extract the image id

    Parameters
    ----------
    static_url : str
        Static url to photo, one of kind:
            https://farm{farm-id}.staticflickr.com/{server-id}/{id}_{secret}.jpg
                or
            https://farm{farm-id}.staticflickr.com/{server-id}/{id}_{secret}_[mstzb].jpg
                or
            https://farm{farm-id}.staticflickr.com/{server-id}/{id}_{o-secret}_o.(jpg|gif|png)

    Returns
    -------
    str
        Image id of url
    """

    pattern = r"(?:.*?\/\/?)+([^_]*)"
    image_id = re.findall(pattern, static_url)[0]
    return image_id


def get_train_val_test_flickr_ids(oiv_folder):
    """ Extract the flickr image ids for train, validation, and test in the OIV dataset

    Parameters
    ----------
    oiv_folder : str
        Path to folder containing open images csv's

    Returns
    -------
    dict of str -> set
        Dict mapping "train", "validation", and "test" to sets, containing the flickr ids of images that belong to them
    """

    files = {"train": "train-images-boxable-with-rotation.csv", "validation": "validation-images-with-rotation.csv",
             "test": "test-images-with-rotation.csv"}
    image_ids = {}
    for subset, file_name in tqdm(files.items()):
        subset_csv = load_csv_as_dict(os.path.join(oiv_folder, file_name), delimiter=",")
        subset_ids = set()
        for row in subset_csv:
            flickr_url = row["OriginalURL"]
            flickr_id = extract_image_id_from_flickr_static(flickr_url)
            subset_ids.add(flickr_id)
        image_ids[subset] = subset_ids
    return image_ids


def get_labels_detected_in_images(oiv_folder):
    """ Extract the labels that are bounding boxes in each image

    Parameters
    ----------
    oiv_folder : str
        Path to folder containing open images csv's

    Returns
    -------
    dict of str -> dict of str -> set
        Dict mapping "train", "validation", and "test" to sets, containing dicts that map flickr_id to set of OIV labels
        that are present in them
    """

    files = [["train", "train-images-boxable-with-rotation.csv", "train-annotations-bbox.csv"],
             ["validation", "validation-images-with-rotation.csv", "validation-annotations-bbox.csv"],
             ["test", "test-images-with-rotation.csv", "test-annotations-bbox.csv"]]

    image_labels = {}
    for subset, image_id_file_name, bbox_file_name in tqdm(files):
        image_id_flickr_id_map = {}
        image_id_file_csv = load_csv_as_dict(os.path.join(oiv_folder, image_id_file_name), delimiter=",")
        for row in image_id_file_csv:
            image_id = row["ImageID"]
            flickr_url = row["OriginalURL"]
            flickr_id = extract_image_id_from_flickr_static(flickr_url)
            image_id_flickr_id_map[image_id] = flickr_id
        subset_image_labels = {}
        bbox_file_csv = load_csv_as_dict(os.path.join(oiv_folder, bbox_file_name), delimiter=",")
        for row in bbox_file_csv:
            image_id = row["ImageID"]
            flickr_id = image_id_flickr_id_map[image_id]
            if flickr_id not in subset_image_labels:
                subset_image_labels[flickr_id] = set()
            label = row["LabelName"]
            subset_image_labels[flickr_id].add(label)
        image_labels[subset] = subset_image_labels
    return image_labels


def _hierarchy_child_to_dict(hierarchy_dict, labels_map=None):
    """ Recursively constructs child dictionaries for children of hierarchy dict

    Parameters
    ----------
    hierarchy_dict : dict
        Dict with key "LabelName", indicating the label in the hierarchy this dict is for, may also have key
        "Subcategory", which contains the children hierarchy_dict's of the node
    labels_map : dict
        Maps OIV labels to OIV names, if not passed then nodes in hierarchy will be labelled with OIV label, otherwise
        their OIV name

    Returns
    -------
    dict of str -> dict of str -> ...
        If hierarchy_dict has children, then returns _hierarchy_child_to_dict of children dicts, otherwise returns None
    """

    if not hierarchy_dict.get("Subcategory"):
        return None
    children = {}
    for child in hierarchy_dict["Subcategory"]:
        child_label = child["LabelName"]
        if labels_map:
            child_label = labels_map[child_label]
        children[child_label] = _hierarchy_child_to_dict(child, labels_map=labels_map)
    return children


def hierachy_to_dict(hierarchy_file, label_names_file=None):
    """ Takes a class hierarchy json and converts it into a more readable dict

    Parameters
    ----------
    hierarchy_file : str
        Path to hierarchy file json
    label_names_file : str
        Path to file mapping OIV labels to OIV names. Default of None. If passed then OIV labels are replaced with
        human readable ones

    Returns
    -------
    dict of str -> dict of str -> ...
        Recursively constructs dict with first level dicts mapping first level concepts in the hierarchy to dicts of
        second level concepts, etc.
    """

    f = open(hierarchy_file, "r")
    hierachy_json = json.load(f)
    labels_map = None
    if label_names_file:
        label_names_csv = load_csv_as_dict(label_names_file, delimiter=",", fieldnames=["oiv_label", "oiv_name"])
        labels_map = {}
        for row in label_names_csv:
            labels_map[row["oiv_label"]] = row["oiv_name"]
    return _hierarchy_child_to_dict(hierachy_json, labels_map)
