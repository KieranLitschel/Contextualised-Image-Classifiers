from common import load_csv_as_dict, write_rows_to_csv
from oiv.common import get_train_val_test_ids, get_labels_detected_in_images
from yfcc100m.common import get_dataset_fields, get_autotag_fields
from oiv.common import get_hierarchy_json_path, hierarchy_members_list, get_hierarchy_classes_parents
from yfcc100m.class_alignment import get_yfcc100m_oiv_labels_map
from embeddings.load import build_features_encoder
from embeddings.encoders import CommaTokenTextEncoder
import os
from tqdm import tqdm
import pandas
import re
import tensorflow as tf
import pycld2 as cld2
from nltk.stem import SnowballStemmer


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
        Tuple separated pre processed user tags
    """

    remove_nums = remove_nums if remove_nums is not None else True
    stem = stem if stem is not None else True
    if remove_nums:
        image_user_tags = ",".join([tag for tag in image_user_tags.split(",") if not re.match(r"^[0-9]+$", tag)])
        if not image_user_tags:
            return ""
    if stem:
        is_reliable, _, details = cld2.detect(image_user_tags)
        language = details[0][0].lower()
        if is_reliable and language != "unknown":
            if language in SnowballStemmer.languages:
                stemmer = SnowballStemmer(language)
                stemmed_user_tags = []
                for user_tag in image_user_tags.split(","):
                    stemmed_tag = "+".join([stemmer.stem(word) if word != '' else '' for word in user_tag.split("+")])
                    stemmed_user_tags.append(stemmed_tag)
                image_user_tags = ",".join(stemmed_user_tags)
    return image_user_tags


def _write_progress_to_csv(output_folder, yfcc_lines, oiv_lines, oiv_human_verified_lines):
    """ Write progress building yfcc, oiv human and machine, and oiv human to subset files

    Parameters
    ----------
    output_folder : str
        Folder to output datasets to
    yfcc_lines : dict of str -> dict of str -> list
        Lines for YFCC training set
    oiv_lines : dict of str -> dict of str -> list
        Lines for OIV training and validation sets with mixtures of machine and human tags
    oiv_human_verified_lines : dict of str -> dict of str -> list
        Lines for OIV validation and test sets with only human tags
    """

    for dataset_folder, lines in zip(["yfcc_lines", "oiv_lines", "oiv_human_verified_lines"],
                                     [yfcc_lines, oiv_lines, oiv_human_verified_lines]):
        dataset_folder_path = os.path.join(output_folder, dataset_folder)
        if not os.path.exists(dataset_folder_path):
            os.makedirs(dataset_folder_path)
        for subset in ["train", "validation", "test"]:
            subset_lines = lines.get(subset)
            if subset_lines:
                csv_path = os.path.join(dataset_folder_path, "{}.csv".format(subset))
                with open(csv_path, "a") as csv:
                    csv.writelines(subset_lines)


def join_dataset_and_autotags(dataset_path, autotags_path, oiv_folder, output_folder, stem=None):
    """ Reads the dataset and autotags files, and writes the id, user tags (stemmed if stemmable language detected), and
        auto tags for each image (discarding of videos) to the datasets subsets at the output path, by appending the
        rows to it. Does so for datasets YFCC, OIV human and machine, and OIV human

    Parameters
    ----------
    dataset_path : str
        Path to dataset file
    autotags_path : str
        Path to autotags file
    oiv_folder : str
        Path to folder of Image ID files of Open Images for train, validation, and test
    output_folder : str
        Folder to output datasets to
    stem : bool
        Whether to apply stemming, default is True
    """

    stem = stem if stem is not None else True
    dataset = load_csv_as_dict(dataset_path, fieldnames=get_dataset_fields())
    autotags = load_csv_as_dict(autotags_path, fieldnames=get_autotag_fields())
    hierarchy_file_path = get_hierarchy_json_path()
    classes_to_keep = set(hierarchy_members_list(hierarchy_file_path))
    classes_parents = get_hierarchy_classes_parents(hierarchy_file_path)
    print("Getting Open Images image IDs")
    oiv_image_ids = get_train_val_test_ids(oiv_folder, flickr_ids=False)
    print("Getting Open Images labels")
    oiv_image_labels = get_labels_detected_in_images(oiv_folder, classes_to_keep=classes_to_keep,
                                                     get_confidence=True)
    yfcc100m_oiv_labels_map = get_yfcc100m_oiv_labels_map()
    yfcc_lines = {"train": []}
    oiv_lines = {"train": [], "validation": []}
    oiv_human_verified_lines = {"validation": [], "test": []}
    lines_in_memory = 0
    print("Building dataset")
    for dataset_row in tqdm(dataset):
        autotags_row = next(autotags)
        image_id = dataset_row["ID"]
        image_user_tags = dataset_row["UserTags"]
        if dataset_row["Video"] == "1":
            continue
        image_user_tags = pre_process_user_tags(image_user_tags, stem=stem)
        if not image_user_tags:
            continue
        if image_id in oiv_image_ids["train"] or image_id in oiv_image_ids["validation"] \
                or image_id in oiv_image_ids["test"]:
            for label, confidence in oiv_image_labels[image_id].items():
                confidence = float(confidence)
                if label in classes_parents:
                    for parent_label in classes_parents[label]:
                        if parent_label in oiv_image_labels[image_id] and \
                                confidence < oiv_image_labels[image_id][parent_label]:
                            continue
                        oiv_image_labels[image_id][parent_label] = confidence
            image_labels = oiv_image_labels[image_id].items()
        elif autotags_row["PredictedConcepts"]:
            image_labels = [tag_prob.split(":") for tag_prob in autotags_row["PredictedConcepts"].split(",")]
            new_image_labels = {}
            for yfcc_tag, confidence in image_labels:
                confidence = float(confidence)
                if yfcc_tag in yfcc100m_oiv_labels_map:
                    oiv_tag = yfcc100m_oiv_labels_map[yfcc_tag]
                    new_image_labels[oiv_tag] = confidence
                    if oiv_tag in classes_parents:
                        for parent_tag in classes_parents[oiv_tag]:
                            if parent_tag in new_image_labels and confidence < new_image_labels[parent_tag]:
                                continue
                            new_image_labels[parent_tag] = confidence
            image_labels = new_image_labels.items()
        else:
            image_labels = []
        if classes_to_keep:
            image_labels = [(tag, confidence) for tag, confidence in image_labels if tag in classes_to_keep]
        image_labels_str = ",".join("{}:{}".format(tag, confidence) for tag, confidence in image_labels)
        line = "{}\t{}\t{}\n".format(image_id, image_user_tags, image_labels_str)
        subset = "train"
        if image_id in oiv_image_ids["validation"]:
            subset = "validation"
        elif image_id in oiv_image_ids["test"]:
            subset = "test"
        if subset == "validation":
            oiv_lines[subset].append(line)
            lines_in_memory += 1
        if subset == "validation" or subset == "test":
            human_image_labels = []
            for tag, confidence in image_labels:
                if confidence == 0 or confidence == 1:
                    human_image_labels.append((tag, confidence))
            if human_image_labels:
                human_image_labels_str = ",".join("{}:{}".format(tag, confidence)
                                                  for tag, confidence in human_image_labels)
                oiv_human_verified_lines[subset].append(human_image_labels_str)
                lines_in_memory += 1
        elif image_id in oiv_image_ids["train"]:
            oiv_lines[subset].append(line)
            yfcc_lines[subset].append(line)
            lines_in_memory += 2
        else:
            yfcc_lines[subset].append(line)
            lines_in_memory += 1
        if lines_in_memory == 10000000:
            _write_progress_to_csv(output_folder, yfcc_lines, oiv_lines, oiv_human_verified_lines)
            yfcc_lines = {"train": []}
            oiv_lines = {"train": [], "validation": []}
            oiv_human_verified_lines = {"validation": [], "test": []}
            lines_in_memory = 0
    _write_progress_to_csv(output_folder, yfcc_lines, oiv_lines, oiv_human_verified_lines)


def build_dataset(dataset_dir, classes_encoder_path, output_folder, tag_threshold, user_tags_limit):
    """ Takes

    Parameters
    ----------
    dataset_dir : str
        Path to folder produced by joined_to_subsets
    classes_encoder_path : str
        Path to classes encoder produced by build_classes_encoder
    output_folder : str
        Folder to output feature encoder and TFRecord folds to
    tag_threshold : int
        Threshold over which to keep words as features. Default of None. If None all are kept
    user_tags_limit : int
        If an image has more user tags than this value, then the tags beyond this value are ignored. Default of None.
        If None all are kept
    """
    classes_encoder = CommaTokenTextEncoder.load_from_file(classes_encoder_path)
    features_encoder = build_features_encoder(os.path.join(dataset_dir, "train"),
                                              tag_threshold=tag_threshold,
                                              user_tags_limit=user_tags_limit)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for subset in ["train", "validation", "test"]:
        print("Building TFRecord for {}".format(subset))
        subset_path = os.path.join(dataset_dir, subset)
        subset_csv = pandas.read_csv(subset_path, sep="\t", na_filter=False).values
        with tf.python_io.TFRecordWriter(os.path.join(output_folder, subset + ".tfrecords")) as writer:
            for row in tqdm(subset_csv):
                flickr_id, user_tags, labels = row
                user_tags = ",".join([tag for tag in user_tags.split(",")][:user_tags_limit])
                labels = ",".join([label_prob.split(":")[0] for label_prob in labels.split(",")])
                encoded_features = [feature for feature in features_encoder.encode(user_tags) if
                                    feature != features_encoder.vocab_size - 1]
                if not encoded_features:
                    continue
                encoded_labels = [label - 1 for label in classes_encoder.encode(labels)]
                example = tf.train.Example(features=tf.train.Features(feature={
                    "flickr_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[flickr_id])),
                    "encoded_features": tf.train.Feature(int64_list=tf.train.Int64List(value=encoded_features)),
                    "encoded_labels": tf.train.Feature(int64_list=tf.train.Int64List(value=encoded_labels)),
                }))
                writer.write(example.SerializeToString())
    features_encoder.save_to_file(os.path.join(output_folder, "features_encoder"))


def count_frequency_of_number_of_user_tags(train_path):
    """ Counts the frequency that each possible number of tags occurs

    Parameters
    ----------
    train_path : str
        Path to training subset produced by joined_to_subsets

    Returns
    -------
    dict of int -> int
        Maps number of tags to its frequency across the training set
    """

    train = load_csv_as_dict(train_path, fieldnames=["ID", "UserTags", "PredictedConcepts"])
    len_freqs = {}
    for row in tqdm(train):
        tags = row["UserTags"].split(",")
        no_tags = len(tags)
        if no_tags not in len_freqs:
            len_freqs[no_tags] = 0
        len_freqs[no_tags] += 1
    return len_freqs
