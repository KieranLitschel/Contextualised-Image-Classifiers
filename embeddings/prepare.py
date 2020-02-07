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


def join_dataset_and_autotags(dataset_path, autotags_path, oiv_folder, yfcc_output_path, oiv_output_path,
                              oiv_human_output_path, stem=None):
    """ Reads the dataset and autotags files, and writes the id, user tags (stemmed if stemmable language detected), and
        auto tags for each image (discarding of videos) to the file at output path, by appending the rows to it

    Parameters
    ----------
    dataset_path : str
        Path to dataset file
    autotags_path : str
        Path to autotags file
    oiv_folder : str
        Path to folder of Image ID files of Open Images for train, validation, and test
    yfcc_output_path : str
        File to append yfcc rows to that do not appear in oiv validation and test
    oiv_output_path : str
        File to append oiv rows to
    oiv_human_output_path : str
        File to append oiv rows with human verified labels to
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
    yfcc_lines = []
    oiv_lines = []
    oiv_human_verified_lines = []
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
        if image_id in oiv_image_ids["validation"] or image_id in oiv_image_ids["test"]:
            oiv_lines.append(line)
            human_image_labels = []
            for tag, confidence in image_labels:
                if confidence == 0 or confidence == 1:
                    human_image_labels.append((tag, confidence))
            if human_image_labels:
                human_image_labels_str = ",".join("{}:{}".format(tag, confidence)
                                                  for tag, confidence in human_image_labels)
                oiv_human_verified_lines.append(human_image_labels_str)
        elif image_id in oiv_image_ids["train"]:
            oiv_lines.append(line)
            yfcc_lines.append(line)
        else:
            yfcc_lines.append(line)
        if len(yfcc_lines) == 10000000:
            output_file = open(yfcc_output_path, "a")
            output_file.writelines(yfcc_lines)
            output_file.close()
            yfcc_lines = []
    if yfcc_lines:
        yfcc_lines[-1] = yfcc_lines[-1][:-1]
        output_file = open(yfcc_output_path, "a")
        output_file.writelines(yfcc_lines)
        output_file.close()
    output_file = open(oiv_output_path, "a")
    output_file.writelines(oiv_lines)
    output_file.close()
    output_file = open(oiv_human_output_path, "a")
    output_file.writelines(oiv_human_verified_lines)
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
    subset_members = get_train_val_test_ids(oiv_folder)
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
