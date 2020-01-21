from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tqdm import tqdm
from functools import partial
from common import load_csv_as_dict


def build_classes_encoder(classes_set):
    """ Takes a set of classes and builds a token text encoder, to convert the str classes to numbers

    Parameters
    ----------
    classes_set : set of str
        Set of classes

    Returns
    -------
    tfds.features.text.TokenTextEncoder
        Encoder for classes
    """

    classes_encoder = tfds.features.text.TokenTextEncoder(classes_set, decode_token_separator=",")
    return classes_encoder


def count_user_tags(subset_path, user_tags_limit=None):
    """ Counts the number of user tags

    Parameters
    ----------
    subset_path : str
        Path to a subset produced by joined_to_subsets
    user_tags_limit : int
        If an image has more user tags than this value, then the tags beyond this value are ignored. Default of None.
        If None all are kept

    Returns
    -------
    dict of str -> int
        Count of each user tag
    """

    subset = load_csv_as_dict(subset_path, fieldnames=["ID", "UserTags", "PredictedConcepts"])
    counts = {}
    for row in tqdm(subset):
        tags = row["UserTags"].split(",")
        for tag in tags[:user_tags_limit]:
            if tag not in counts:
                counts[tag] = 0
            counts[tag] += 1
    return counts


def build_features_encoder(subset_path, tag_threshold=None, user_tags_limit=None):
    """ Takes a set of classes and builds a token text encoder, to convert the str classes to numbers

    Parameters
    ----------
    subset_path : str
        Path to subset to build encoder from
    tag_threshold : int
        Threshold over which to keep words as features. Default of None. If None all are kept
    user_tags_limit : int
        If an image has more user tags than this value, then the tags beyond this value are ignored. Default of None.
        If None all are kept

    Returns
    -------
    tfds.features.text.TokenTextEncoder
        Encoder for features
    """

    tag_threshold = tag_threshold or 1
    vocab_count = count_user_tags(subset_path, user_tags_limit=user_tags_limit)
    vocab_list = []
    for vocab, count in vocab_count.items():
        if tag_threshold >= count:
            vocab_list.append(vocab)
    features_encoder = tfds.features.text.TokenTextEncoder(vocab_list, decode_token_separator=",")
    return features_encoder


def _str_row_to_tf(row, user_tags_limit, features_encoder, classes_encoder):
    """ Converts comma separated user tags to list of encoded tag id's, and comma separated classes to one hot encoded
        classes

    Parameters
    ----------
    row : tf.Tensor
        Row from file produced by yfcc100m.embeddings.prepare.joined_to_subsets
    user_tags_limit : int
        Number of user tags to keep
    features_encoder : tfds.features.text.TokenTextEncoder
        User tags encoder
    classes_encoder : tfds.features.text.TokenTextEncoder
        Labels encoder

    Returns
    -------
    tf.int32, tf.bool

    """

    _, user_tags, labels = bytes.decode(row.numpy()).split("\t")
    user_tags = ",".join([tag for tag in user_tags.split(",")][:user_tags_limit])
    labels = ",".join([label_prob.split(":")[0] for label_prob in labels.split(",")])
    encoded_features = features_encoder.encode(user_tags)
    encoded_labels = classes_encoder.encode(labels)
    one_hot_labels = np.zeros(classes_encoder.vocab_size)
    for label_num in encoded_labels:
        one_hot_labels[label_num - 1] = 1
    return tf.cast(encoded_features, tf.int32), tf.convert_to_tensor(one_hot_labels, dtype=tf.bool)


def load_subset_as_tf_data(path, classes_encoder, features_encoder):
    """ Loads the subset passed, encodes the features, and one hot-encodes the classes

    Parameters
    ----------
    path : tf.string
        Path to subset to be loaded
    classes_encoder : tfds.features.text.TokenTextEncoder
        Labels encoder
    features_encoder : tfds.features.text.TokenTextEncoder
        User tags encoder

    Returns
    -------
    tf.python.data.ops.dataset_ops.DatasetV1Adapter
        The subset ready for use in TensorFlow
    """

    raw_dataset = tf.data.TextLineDataset(path)
    custom_str_row_to_tf = partial(_str_row_to_tf, features_encoder=features_encoder, classes_encoder=classes_encoder)
    features_labels_dataset = raw_dataset.map(
        lambda row: tf.py_function(custom_str_row_to_tf, inp=[row], Tout=[tf.int32, tf.bool]))
    return features_labels_dataset


def load_train_val(dataset_folder, classes_encoder, features_encoder):
    """ For train and validation, loads them, encodes the features, and one hot-encodes the classes. Validation dataset
        features encoded using encoder built from train

    Parameters
    ----------
    dataset_folder : str
        The location of the train and validation files
    classes_encoder : tfds.features.text.TokenTextEncoder
        Encoder to convert class labels to numbers
    features_encoder : tfds.features.text.TokenTextEncoder
        Encoder to build features from

    Returns
    -------
    tf.python.data.ops.dataset_ops.DatasetV1Adapter, tf.python.data.ops.dataset_ops.DatasetV1Adapter
        First element is the train data, second is the validation data
    """

    train_dataset = load_subset_as_tf_data(os.path.join(dataset_folder, "train"), classes_encoder,
                                           features_encoder=features_encoder)
    val_dataset = load_subset_as_tf_data(os.path.join(dataset_folder, "validation"), classes_encoder,
                                         features_encoder=features_encoder)
    return train_dataset, val_dataset
