import pickle
import re
import urllib

import cld3
import pycld2 as cld2
from tqdm import tqdm

from common import load_csv_as_dict
from embeddings.prepare import pre_process_user_tags
from oiv.common import get_train_val_test_ids
from yfcc100m.common import get_dataset_fields


def count_user_tags(path, stem=None, remove_nums=None, oiv_folder=None):
    """ Count the number of times each user tag occurs in OIV training set

    Parameters
    ----------
    path : str
        Path to dataset file
    remove_nums : bool
        Whether to remove user tags that are only numbers, default True
    stem : bool
        Whether to stem user tags based on their detected language, default True
    oiv_folder : str
        Path to OIV folder. If specified then will only count user tags for OIV training subset

    Returns
    -------
    dict of str -> int
        Number of occurrences of each user tag
    """

    stem = stem if stem is not None else True
    remove_nums = remove_nums if remove_nums is not None else True
    dataset = load_csv_as_dict(path, fieldnames=get_dataset_fields())
    oiv_train_image_ids = {}
    if oiv_folder:
        print("Getting ids of OIV images")
        oiv_train_image_ids = get_train_val_test_ids(oiv_folder)["train"]
    print("Counting occurrences of user tags")
    tag_counts = {}
    for row in tqdm(dataset):
        if not oiv_train_image_ids.get(row["ID"]):
            continue
        user_tags = row["UserTags"]
        if user_tags:
            if stem or remove_nums:
                user_tags = pre_process_user_tags(user_tags, stem=stem, remove_nums=remove_nums)
            if not user_tags:
                continue
            tags = user_tags.split(",")
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


def count_detected_languages_cld2(yfcc_train, keep_numbers=None):
    """ Counts detected languages across YFCC100M using cld2

    Parameters
    ----------
    yfcc_train : str
        Path to YFCC train file produced by join_dataset_and_autotags
    keep_numbers : bool
        Whether to keep numbers, default False

    Returns
    -------
    dict of int -> int
        Maps language to its detected frequency across YFCC100M
    """

    keep_numbers = keep_numbers if keep_numbers is not None else False
    dataset = load_csv_as_dict(yfcc_train, fieldnames=["ImageID", "UserTags", "Classes"])
    language_counts = {}
    for dataset_row in tqdm(dataset):
        image_user_tags = dataset_row["UserTags"]
        pre_processed_image_user_tags = pre_process_user_tags(image_user_tags, remove_nums=not keep_numbers, stem=False)
        decoded_pre_processed_image_user_tags = ''.join(
            x for x in urllib.parse.unquote(re.sub(r"[,+]", " ", pre_processed_image_user_tags)) if x.isprintable())
        if not image_user_tags:
            continue
        is_reliable, _, details = cld2.detect(decoded_pre_processed_image_user_tags)
        language = details[0][0].lower()
        if not is_reliable:
            language = "unknown"
        if not language_counts.get(language):
            language_counts[language] = 0
        language_counts[language] += 1
    return language_counts


def count_detected_languages_cld3(yfcc_train, keep_numbers=None):
    """ Counts detected languages across YFCC100M using cld3

    Parameters
    ----------
    yfcc_train : str
        Path to YFCC train file produced by join_dataset_and_autotags
    keep_numbers : bool
        Whether to keep numbers, default False

    Returns
    -------
    dict of int -> int
        Maps language to its detected frequency across YFCC100M
    """

    keep_numbers = keep_numbers if keep_numbers is not None else False
    dataset = load_csv_as_dict(yfcc_train, fieldnames=["ImageID", "UserTags", "Classes"])
    language_counts = {}
    for dataset_row in tqdm(dataset):
        image_user_tags = dataset_row["UserTags"]
        pre_processed_image_user_tags = pre_process_user_tags(image_user_tags, remove_nums=not keep_numbers, stem=False)
        decoded_pre_processed_image_user_tags = ''.join(
            x for x in urllib.parse.unquote(re.sub(r"[,+]", " ", pre_processed_image_user_tags)) if x.isprintable())
        if not image_user_tags:
            continue
        image_user_tags = re.sub(r"\b(?:https?://|www\.)[a-z0-9-]+(\.[a-z0-9-]+)+(?:[/?].*)?", "",
                                 decoded_pre_processed_image_user_tags)
        lp = cld3.get_language(image_user_tags)
        if lp:
            lang_code = lp.language
            is_reliable = lp.is_reliable
            if not is_reliable:
                lang_code = "unknown"
        else:
            lang_code = "unknown"
        if lang_code not in language_counts:
            language_counts[lang_code] = 0
        language_counts[lang_code] += 1
    return language_counts
