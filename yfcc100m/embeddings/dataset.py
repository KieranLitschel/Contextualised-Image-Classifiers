from yfcc100m.load import load_csv_as_dict
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
                tag = tag.lower()
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


def dataset_to_file(dataset_path, autotags_path, output_path):
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
    """

    dataset = load_csv_as_dict(dataset_path, fieldnames=FIELDS)
    autotags = load_csv_as_dict(autotags_path, fieldnames=["ID", "PredictedConcepts"])
    lines = []
    for dataset_row in tqdm(dataset):
        autotags_row = next(autotags)
        image_id = dataset_row["ID"]
        image_user_tags = dataset_row["UserTags"]
        image_auto_tags = autotags_row["PredictedConcepts"]
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
