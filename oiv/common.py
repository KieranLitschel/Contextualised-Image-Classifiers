import re
import os
from common import load_csv_as_dict, write_rows_to_csv
from tqdm import tqdm
import json
import pathlib


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


def get_train_val_test_ids(oiv_folder, flickr_ids=None):
    """ Extract the ids for train, validation, and test in the OIV dataset. If flickr_ids set to True, they are converted
        to flickr_ids. Default of True

    Parameters
    ----------
    oiv_folder : str
        Path to folder containing open images csv's
    flickr_ids : bool
        Whether to return flickr ids. Default of True. Otherwise returns OIV ids

    Returns
    -------
    dict of str -> set
        Dict mapping "train", "validation", and "test" to sets, containing the flickr ids of images that belong to them
    """

    flickr_ids = flickr_ids if flickr_ids is not None else True
    files = {"train": "train-images-with-labels-with-rotation.csv", "validation": "validation-images-with-rotation.csv",
             "test": "test-images-with-rotation.csv"}
    image_ids = {}
    for subset, file_name in tqdm(files.items()):
        subset_csv = load_csv_as_dict(os.path.join(oiv_folder, file_name), delimiter=",")
        subset_ids = set()
        for row in subset_csv:
            if flickr_ids:
                flickr_url = row["OriginalURL"]
                flickr_id = extract_image_id_from_flickr_static(flickr_url)
                subset_ids.add(flickr_id)
            else:
                subset_ids.add(row["ImageID"])
        image_ids[subset] = subset_ids
    return image_ids


def get_labels_detected_in_images(oiv_folder, classes_to_keep=None, get_confidence=None):
    """ Extract the labels that are bounding boxes in each image

    Parameters
    ----------
    oiv_folder : str
        Path to folder containing open images csv's
    classes_to_keep : set
        Set of classes to keep. If None all classes kept
    get_confidence : bool
        Whether to return the confidence. If False then just returns the labels for each image as a set, otherwise
        returns them as a dictionary with the labels as keys and confidences as values. Default of False

    Returns
    -------
    dict of str -> dict of str -> set
        Dict mapping "train", "validation", and "test" to sets, containing dicts that map flickr_id to set of OIV labels
        that are present in them. If get_confidence is True then returns dict of str -> float as described above
    """

    get_confidence = get_confidence if get_confidence is not None else False
    files = [["train", "train-images-with-labels-with-rotation.csv", "train-annotations-human-machine-imagelabels.csv"],
             ["validation", "validation-images-with-rotation.csv", "validation-annotations-human-machine-imagelabels.csv"],
             ["test", "test-images-with-rotation.csv", "test-annotations-human-imagelabels.csv"]]

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
            label = row["LabelName"]
            confidence = row["Confidence"]
            if classes_to_keep and label not in classes_to_keep:
                continue
            if confidence == "0":
                continue
            if flickr_id not in subset_image_labels:
                subset_image_labels[flickr_id] = set() if not get_confidence else {}
            if not get_confidence:
                subset_image_labels[flickr_id].add(label)
            else:
                subset_image_labels[flickr_id][label] = float(confidence)
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


def hierarchy_members_list(hierarchy_file, label_names_file=None):
    """ Takes a class hierarchy json and returns the list of classes in the hierarchy

    Parameters
    ----------
    hierarchy_file : str
        Path to hierarchy file json
    label_names_file : str
        Path to file mapping OIV labels to OIV names. Default of None. If passed then OIV labels are replaced with
        human readable ones

    Returns
    -------
    list of str
        List of OIV labels in the hierarchy, or names if label_names_file is passed
    """

    members = []
    hierarchy_dict_queue = [hierachy_to_dict(hierarchy_file, label_names_file=label_names_file)]
    while hierarchy_dict_queue:
        hierarchy_dict = hierarchy_dict_queue.pop(0)
        for member in hierarchy_dict.keys():
            members.append(member)
        for child in hierarchy_dict.values():
            if child:
                hierarchy_dict_queue.append(child)
    return members


def get_hierarchy_json_path():
    """ Get the path to the hierarchy JSON

    Returns
    -------
    str
        Path to hierarchy json
    """

    return os.path.join(pathlib.Path(__file__).parent.absolute(), "challenge-2019-label500-hierarchy.json")


def _get_hierarchy_classes_parents(hierarchy_dict, classes_parents, curr_parents):
    """ Gets a dict that contains the ascendant in the hierarchy of each label. If a label has no ascendants its list is
        empty

    Parameters
    ----------
    hierarchy_dict : dict of str -> dict of str -> ...
        Recursively constructed dict with first level dicts mapping first level concepts in the hierarchy to dicts of
        second level concepts, etc.
    classes_parents : dict of str -> list of str
        Maps each label to a list of its parents
    curr_parents : list of str
        List of ascendants of the current hierarchy_dicts label

    Returns
    -------
    dict of str -> list of str
        Maps each label to a list of its parents
    """

    for label, child_hierarchy_dict in hierarchy_dict.items():
        classes_parents[label] = curr_parents.copy()
        if child_hierarchy_dict:
            curr_parents.append(label)
            classes_parents = _get_hierarchy_classes_parents(child_hierarchy_dict, classes_parents, curr_parents)
            curr_parents.pop(-1)
    return classes_parents


def get_hierarchy_classes_parents(hierarchy_file, label_names_file=None):
    """ Gets a dict that contains the ascendant in the hierarchy of each label. If a label has no ascendants its list is
        empty

    Parameters
    ----------
    hierarchy_file : str
        Path to hierarchy file json
    label_names_file : str
        Path to file mapping OIV labels to OIV names. Default of None. If passed then OIV labels are replaced with
        human readable ones

    Returns
    -------
    dict of str -> list of str
        Maps each label to a list of its parents
    """

    hierarchy_dict = hierachy_to_dict(hierarchy_file, label_names_file=label_names_file)
    classes_parents = _get_hierarchy_classes_parents(hierarchy_dict, {}, [])
    return classes_parents


def build_human_machine_labels(oiv_folder):
    """ Reads the image level image id and human label files of OIV joined with YFCC100M by OpenImagesV5Tools (so that
        they only contain OIV images that overlap with YFCC100M) for train and validation, and  makes a new file
        containing the human and machine labels. Where an image has a label from both the human and  machine dataset,
        machine label is discarded

    Parameters
    ----------
    oiv_folder : str
        Path to folder containing open images csv's
    """

    valid_image_ids = set.union(*get_train_val_test_ids(oiv_folder, flickr_ids=False).values())
    for subset in ["train", "validation"]:
        print("Building human-machine labels for {}".format(subset))
        human_labels_filename = "{}-annotations-human-imagelabels.csv".format(subset)
        machine_labels_filename = "{}-annotations-machine-imagelabels.csv".format(subset)
        human_machine_labels_file = "{}-annotations-human-machine-imagelabels.csv".format(subset)
        human_labels = {}
        human_labels_csv = load_csv_as_dict(os.path.join(oiv_folder, human_labels_filename), delimiter=",")
        new_rows = []
        for row in tqdm(human_labels_csv):
            image_id = row["ImageID"]
            if image_id not in valid_image_ids:
                continue
            label = row["LabelName"]
            if image_id not in human_labels:
                human_labels[image_id] = set()
            human_labels[image_id].add(label)
            new_rows.append(row)
        write_rows_to_csv(new_rows, os.path.join(oiv_folder, human_machine_labels_file), mode="a", delimiter=",",
                          write_header=True)
        new_rows = []
        machine_labels_csv = load_csv_as_dict(os.path.join(oiv_folder, machine_labels_filename), delimiter=",")
        for row in tqdm(machine_labels_csv):
            image_id = row["ImageID"]
            if image_id not in valid_image_ids:
                continue
            label = row["LabelName"]
            if image_id not in human_labels or label in human_labels[image_id]:
                continue
            new_rows.append(row)
            if len(new_rows) == 10000000:
                write_rows_to_csv(new_rows, os.path.join(oiv_folder, human_machine_labels_file), mode="a",
                                  delimiter=",")
                new_rows = []
        if len(new_rows) != 0:
            write_rows_to_csv(new_rows, os.path.join(oiv_folder, human_machine_labels_file), mode="a", delimiter=",")
