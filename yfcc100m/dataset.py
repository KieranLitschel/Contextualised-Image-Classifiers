from common import load_csv_as_dict
from yfcc100m.common import get_dataset_fields
import pickle
from tqdm import tqdm
import pycld2 as cld2


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

    dataset = load_csv_as_dict(path, fieldnames=get_dataset_fields())
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
    dataset = load_csv_as_dict(path, fieldnames=get_dataset_fields())
    highest_counts = {}
    for row in tqdm(dataset):
        image_id = row["ID"]
        count = 0
        if row["UserTags"]:
            user_tags = row["UserTags"].split(",")
            count = max(tag_counts[user_tag] for user_tag in user_tags)
        highest_counts[image_id] = count
    return highest_counts


def count_detected_languages(dataset_path, keep_numbers=None):
    """ Counts detected languages across YFCC100M

    Parameters
    ----------
    dataset_path : str
        Path to dataset file
    keep_numbers : bool
        Whether to keep numbers, default False

    Returns
    -------
    dict of int -> int
        Maps language to its detected frequency across YFCC100M
    """

    keep_numbers = keep_numbers if keep_numbers is not None else False
    dataset = load_csv_as_dict(dataset_path, fieldnames=get_dataset_fields())
    languages_count = {}
    for dataset_row in tqdm(dataset):
        image_user_tags = dataset_row["UserTags"]
        if dataset_row["Video"] == "1":
            continue
        if not keep_numbers:
            image_user_tags = ",".join([tag for tag in image_user_tags.split(",") if not re.match(r"^[0-9]+$", tag)])
        if not image_user_tags:
            continue
        is_reliable, _, details = cld2.detect(image_user_tags)
        language = details[0][0].lower()
        if not is_reliable:
            language = "unknown"
        if language not in languages_count:
            languages_count[language] = 0
        languages_count[language] += 1
    return languages_count
