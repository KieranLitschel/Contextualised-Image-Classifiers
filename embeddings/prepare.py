import os
import re
import urllib

import pycld2 as cld2
from nltk.stem import SnowballStemmer
from tqdm import tqdm

from common import load_csv_as_dict
from oiv.common import get_hierarchy_json_path, hierarchy_members_list, get_hierarchy_classes_parents
from oiv.common import get_train_val_test_ids, get_labels_detected_in_images
from yfcc100m.class_alignment import get_yfcc100m_oiv_labels_map
from yfcc100m.common import get_dataset_fields, get_autotag_fields
from yfcc100m.dataset import detect_language_cld2, decode_image_user_tags


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

    for dataset_folder, lines in zip(["yfcc", "oiv", "oiv_human_verified"],
                                     [yfcc_lines, oiv_lines, oiv_human_verified_lines]):
        dataset_folder_path = os.path.join(output_folder, dataset_folder)
        if not os.path.exists(dataset_folder_path):
            os.makedirs(dataset_folder_path)
        for subset in ["train", "validation", "test"]:
            subset_lines = lines.get(subset)
            if subset_lines:
                csv_path = os.path.join(dataset_folder_path, "{}.tsv".format(subset))
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
    oiv_image_ids = get_train_val_test_ids(oiv_folder, flickr_ids=True)
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
        subset = "train"
        if image_id in oiv_image_ids["validation"]:
            subset = "validation"
        elif image_id in oiv_image_ids["test"]:
            subset = "test"
        image_user_tags = dataset_row["UserTags"]
        if dataset_row["Video"] == "1":
            continue
        image_user_tags = pre_process_user_tags(image_user_tags, stem=stem)
        if not image_user_tags:
            continue
        if image_id in oiv_image_ids["train"] or subset == "validation" or subset == "test":
            if oiv_image_labels[subset].get(image_id):
                parent_labels = {}
                for label, confidence in oiv_image_labels[subset][image_id].items():
                    confidence = float(confidence)
                    if label in classes_parents:
                        for parent_label in classes_parents[label]:
                            if parent_labels.get(parent_label) and confidence < parent_labels[parent_label]:
                                continue
                            parent_labels[parent_label] = confidence
                for parent_label, confidence in parent_labels.items():
                    oiv_image_labels[subset][image_id][parent_label] = confidence
                image_labels = oiv_image_labels[subset][image_id].items()
            else:
                image_labels = []
        elif autotags_row["PredictedConcepts"]:
            image_labels = [tag_prob.split(":") for tag_prob in autotags_row["PredictedConcepts"].split(",")]
            new_image_labels = {}
            for yfcc_tag, confidence in image_labels:
                confidence = float(confidence)
                if yfcc100m_oiv_labels_map.get(yfcc_tag):
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
        image_labels = [(tag, confidence) for tag, confidence in image_labels if tag in classes_to_keep]
        image_labels_str = ",".join("{}:{}".format(tag, confidence) for tag, confidence in image_labels)
        line = "{}\t{}\t{}\n".format(image_id, image_user_tags, image_labels_str)
        if subset == "validation" or subset == "test":
            human_image_labels = []
            for tag, confidence in image_labels:
                if confidence == 0 or confidence == 1:
                    human_image_labels.append((tag, confidence))
            if human_image_labels:
                if subset == "validation":
                    oiv_lines[subset].append(line)
                    lines_in_memory += 1
                human_image_labels_str = ",".join("{}:{}".format(tag, confidence)
                                                  for tag, confidence in human_image_labels)
                human_image_labels_line = "{}\t{}\t{}\n".format(image_id, image_user_tags, human_image_labels_str)
                oiv_human_verified_lines[subset].append(human_image_labels_line)
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


def build_dataset(dataset_dir, classes_encoder_path, features_encoder_path, output_folder):
    """ Builds the tsv (tab-separated values) dataset into TFRecords

    Parameters
    ----------
    dataset_dir : str
        Path to dataset produced by joined_to_subsets
    classes_encoder_path : str
        Path to classes encoder produced by build_classes_encoder
    features_encoder_path : str
        Path to features encoder produced by build_features_encoder
    output_folder : str
        Folder to output feature encoder and TFRecord folds to
    """

    # this code is commented out as it is untested, I'll need to revisit this method when implementing pre-training
    # on YFCC. It isn't necessary for OIV, as the files are so small they fit into memory
    """
    classes_encoder = CommaTokenTextEncoder.load_from_file(classes_encoder_path)
    features_encoder = CommaTokenTextEncoder.load_from_file(features_encoder_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for subset in ["train", "validation", "test"]:
        subset_path = os.path.join(dataset_dir, subset)
        if not os.path.exists(subset_path):
            continue
        print("Building TFRecord for {}".format(subset))
        subset_csv = pandas.read_csv(subset_path, sep="\t", na_filter=False).values
        with tf.python_io.TFRecordWriter(os.path.join(output_folder, subset + ".tfrecords")) as writer:
            for row in tqdm(subset_csv):
                flickr_id, user_tags, labels = row
                labels = ",".join([label_prob.split(":")[0] for label_prob in labels.split(",")])
                confidences = ",".join([label_prob.split(":")[1] for label_prob in labels.split(",")])
                encoded_features = features_encoder.encode(user_tags)
                # get rid of the label number in the encoder (0) reserved for padding
                encoded_labels = [label - 1 for label in classes_encoder.encode(labels)]
                sparse_labels = tf.SparseTensor(indices=encoded_labels, values=confidences,
                                                dense_shape=[classes_encoder.vocab_size - 2])
                example = tf.train.Example(features=tf.train.Features(feature={
                    "flickr_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[flickr_id])),
                    "encoded_features": tf.train.Feature(int64_list=tf.train.Int64List(value=encoded_features)),
                    "encoded_labels": tf.train.Feature(bytes_list=
                                                       tf.train.BytesList(value=tf.serialize_tensor(sparse_labels))),
                }))
                writer.write(example.SerializeToString())
    """
    raise NotImplemented


def count_frequency_of_number_of_user_tags(train_path):
    """ Counts the frequency that each possible number of tags occurs

    Parameters
    ----------
    train_path : str
        Path to training subset produced by join_dataset_and_autotags

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


def count_classes_in_subset(subset_path):
    """ Counts the frequency of classes in the subset

    Parameters
    ----------
    subset_path : str
        Path to training subset produced by join_dataset_and_autotags

    Returns
    -------
    dict of str -> int
        Maps class label to frequency in subset
    """

    subset = load_csv_as_dict(subset_path, fieldnames=["ID", "UserTags", "PredictedConcepts"], delimiter="\t")
    classes = {}
    for row in subset:
        if not row["PredictedConcepts"]:
            continue
        for label_confidence in row["PredictedConcepts"].split(","):
            label, _ = label_confidence.split(":")
            if not classes.get(label):
                classes[label] = 0
            classes[label] += 1
    return classes
