from yfcc100m.common import load_csv_as_dict
from yfcc100m.embeddings.autotags import kept_classes
import pickle
from tqdm import tqdm

FIELDS = ["LineNumber", "ID", "Hash", "UserNSID", "UserNickname", "DateTaken", "DateUploaded",
          "CaptureDevice", "Title", "Description", "UserTags", "MachineTags", "Longitude", "Latitude",
          "LongLatAcc", "PageURL", "DownloadURL", "LicenseName", "LicenseURL", "ServerIdentifier",
          "FarmIdentifier", "Secret", "OriginalSecret", "OriginalExtension", "Video"]


def count_user_tags(path):
    """ Count the number of times each user tag occurs

    Parameters
    ----------
    path : str
        Path to dataset file

    Returns
    -------
    dict of str -> int
        Number of occurrences of each user tag
    """

    dataset = load_csv_as_dict(path, fieldnames=FIELDS)
    tag_counts = {}
    for row in tqdm(dataset):
        if row["UserTags"]:
            tags = row["UserTags"].split(",")
            for tag in tags:
                if not tag_counts.get(tag):
                    tag_counts[tag] = 0
                tag_counts[tag] += 1
    return tag_counts


def images_highest_count_user_tag(path, tag_counts_path=None):
    """ For each image, return the frequency (across the whole dataset) of its tag that occurs
        most often across the dataset

    Parameters
    ----------
    path : str
        Path to dataset file
    tag_counts_path : str
        Path to pickled dict produced by count_user_tags

    Returns
    -------
    dict of str -> int
        Number of occurrences of most frequent user tag for each image
    """

    tag_counts = pickle.load(open(tag_counts_path, "rb")) if tag_counts_path else count_user_tags(path)
    dataset = load_csv_as_dict(path, fieldnames=FIELDS)
    highest_counts = {}
    for row in tqdm(dataset):
        image_id = row["ID"]
        count = 0
        if row["UserTags"]:
            user_tags = row["UserTags"].split(",")
            count = max(tag_counts[user_tag] for user_tag in user_tags)
        highest_counts[image_id] = count
    return highest_counts


def dataset_to_file(dataset_path, autotags_path, output_path, tag_freq_thresh=None, tag_counts_path=None,
                    class_path=None):
    """ Reads the dataset and autotags files, and writes the id, user tags, and auto tags
        for each image to the file at output path, by appending the rows to it

    Parameters
    ----------
    dataset_path : str
        Path to dataset file
    autotags_path : str
        Path to autotags file
    output_path : str
        File to append rows to
    tag_freq_thresh : int
        User tag frequency threshold for keeping. If not specified all user tags kept
    tag_counts_path : str
        Path to pickled dict produced by count_user_tags. If not specified and tag_freq_thresh is not None, calculated
        by reading the dataset
    class_path : str
        Path to classes to keep, to be loaded using kept_classes method. If not specified all classes kept
    """

    dataset = load_csv_as_dict(dataset_path, fieldnames=FIELDS)
    autotags = load_csv_as_dict(autotags_path, fieldnames=["ID", "PredictedConcepts"])
    classes_to_keep = set(kept_classes(class_path)) if class_path else None
    tag_counts = None
    if tag_freq_thresh:
        tag_counts = pickle.load(open(tag_counts_path, "rb")) if tag_counts_path else count_user_tags(dataset_path)
    lines = []
    for dataset_row in tqdm(dataset):
        autotags_row = next(autotags)
        image_id = dataset_row["ID"]
        image_user_tags = dataset_row["UserTags"]
        if tag_counts and image_user_tags:
            image_user_tags = ",".join(
                [tag for tag in image_user_tags.split(",") if tag_counts[tag] >= tag_freq_thresh])
        if not image_user_tags:
            continue
        image_auto_tags = autotags_row["PredictedConcepts"]
        if classes_to_keep and image_auto_tags:
            image_auto_tags = [tag_prob for tag_prob in image_auto_tags.split(",") if
                               tag_prob.split(":")[0] in classes_to_keep]
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
