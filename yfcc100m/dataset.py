import pickle
import re
import urllib

import pycld2 as cld2
from nltk.stem import SnowballStemmer
import cld3
from tqdm import tqdm

from common import load_csv_as_dict
from oiv.common import get_train_val_test_ids
from yfcc100m.common import get_dataset_fields


def pre_process_user_tags(image_user_tags, remove_nums=None, stem=None):
    """ Pre processes user tags, stemming and removing numbers (unless specified)

    Parameters
    ----------
    image_user_tags : str
        Tuple separated user tags
    remove_nums : bool
        Whether to remove user tags that are only numbers, default True
    stem : bool
        Whether to stem user tags based on their detected language, default True

    Returns
    -------
    str
        Comma separated pre processed user tags
    """

    remove_nums = remove_nums if remove_nums is not None else True
    stem = stem if stem is not None else True
    if remove_nums:
        image_user_tags = ",".join([tag for tag in image_user_tags.split(",") if not re.match(r"^[0-9]+$", tag)])
        if not image_user_tags:
            return ""
    if stem:
        is_reliable, language = detect_language_cld2(image_user_tags)
        image_user_tags = decode_image_user_tags(image_user_tags)
        if is_reliable and language != "unknown":
            if language in SnowballStemmer.languages:
                stemmer = SnowballStemmer(language)
                stemmed_user_tags = []
                for user_tag in image_user_tags.split(","):
                    stemmed_tag = "+".join([urllib.parse.quote(stemmer.stem(urllib.parse.unquote(word)))
                                            if word != '' else '' for word in user_tag.split("+")])
                    stemmed_user_tags.append(stemmed_tag)
                image_user_tags = ",".join(stemmed_user_tags)
    return image_user_tags


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
                tag = ''.join(x for x in urllib.parse.unquote(re.sub(r"\+", " ", tag)) if x.isprintable())
                if not tag:
                    continue
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


def decode_image_user_tags(image_user_tags):
    """ Percent decodes comma seperated user tags

    Parameters
    ----------
    image_user_tags : str
        Percent encoded image user tags, with spaces replaced by "+", and user tags comma separated

    Returns
    -------
    str
        User tags with commas and plus replaced by space and then percent decoded
    """

    return ''.join(
        x for x in urllib.parse.unquote(re.sub(r"[,+]", " ", image_user_tags)) if x.isprintable())


def detect_language_cld2(image_user_tags):
    """ Detects the language of the image user tags and whether the detection is reliable

    Parameters
    ----------
    image_user_tags : str
        Percent encoded image user tags, with spaces replaced by "+", and user tags comma separated

    Returns
    -------
    (bool, str)
        First element in tuple says whether the language detection is reliable, second element says the detected
        language
    """

    decoded_pre_processed_image_user_tags = decode_image_user_tags(image_user_tags)
    is_reliable, _, details = cld2.detect(decoded_pre_processed_image_user_tags)
    language = details[0][0].lower()
    return is_reliable, language


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
        is_reliable, language = detect_language_cld2(pre_processed_image_user_tags)
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
        decoded_pre_processed_image_user_tags = decode_image_user_tags(pre_processed_image_user_tags)
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
