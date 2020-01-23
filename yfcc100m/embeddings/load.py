from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
from tqdm import tqdm
from functools import partial
from common import load_csv_as_dict
from yfcc100m.embeddings.encoders import CommaTokenTextEncoder


def build_classes_encoder(classes_set):
    """ Takes a set of classes and builds a token text encoder, to convert the str classes to numbers

    Parameters
    ----------
    classes_set : set of str
        Set of classes

    Returns
    -------
    CommaTokenTextEncoder
        Encoder for classes
    """

    classes_encoder = CommaTokenTextEncoder(classes_set, decode_token_separator=",")
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
    CommaTokenTextEncoder
        Encoder for features
    """

    tag_threshold = tag_threshold or 1
    vocab_count = count_user_tags(subset_path, user_tags_limit=user_tags_limit)
    vocab_list = []
    for vocab, count in vocab_count.items():
        if tag_threshold >= count:
            vocab_list.append(vocab)
    features_encoder = CommaTokenTextEncoder(vocab_list, decode_token_separator=",")
    return features_encoder


def _str_row_to_tf(proto, no_classes):
    """ Converts comma separated user tags to list of encoded tag id's, and comma separated classes to one hot encoded
        classes

    Parameters
    ----------
    proto : Serialized tf.Tensor
        Serialized row from file produced by yfcc100m.embeddings.prepare.subsets_to_tfrecords
    no_classes : int
        Number of classes

    Returns
    -------
    tf.int32, tf.bool
        First element is features, second element is one hot labels
    """

    parsed_features = tf.io.parse_single_example(proto, {
        "encoded_features": tf.io.FixedLenSequenceFeature([], tf.int32, allow_missing=True),
        "encoded_labels": tf.io.FixedLenSequenceFeature([], tf.bool, allow_missing=True),
    })
    encoded_features = bytes.decode(parsed_features["UserTags"].numpy()[0])
    encoded_labels = bytes.decode(parsed_features["PredictedConcepts"].numpy()[0])
    one_hot_labels = tf.reduce_sum(tf.one_hot(indices=encoded_labels, depth=no_classes), reduction_indices=0)
    return tf.cast(encoded_features, tf.int32), tf.convert_to_tensor(one_hot_labels, dtype=tf.bool)


def load_subset_as_tf_data(path, no_classes):
    """ Loads the subset passed, encodes the features, and one hot-encodes the classes

    Parameters
    ----------
    path : tf.string
        Path to subset to be loaded
    no_classes : int
        Number of classes

    Returns
    -------
    tf.python.data.ops.dataset_ops.DatasetV1Adapter
        The subset ready for use in TensorFlow
    """

    raw_dataset = tf.data.TFRecordDataset(path)
    custom_str_row_to_tf = partial(_str_row_to_tf, no_classes=no_classes)
    features_labels_dataset = raw_dataset.map(
        lambda proto: tf.py_function(custom_str_row_to_tf, inp=[proto], Tout=[tf.int32, tf.bool]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    filtered_features_labels_dataset = features_labels_dataset.filter(
        lambda features, _: tf.not_equal(tf.size(features), 0))
    return filtered_features_labels_dataset


def load_train_val(dataset_folder, no_classes):
    """ For train and validation, loads them, encodes the features, and one hot-encodes the classes. Validation dataset
        features encoded using encoder built from train

    Parameters
    ----------
    dataset_folder : str
        The location of the train and validation files
    no_classes : int
        Number of classes

    Returns
    -------
    tf.python.data.ops.dataset_ops.DatasetV1Adapter, tf.python.data.ops.dataset_ops.DatasetV1Adapter
        First element is the train data, second is the validation data
    """

    train_dataset = load_subset_as_tf_data(os.path.join(dataset_folder, "train.tfrecords"), no_classes)
    val_dataset = load_subset_as_tf_data(os.path.join(dataset_folder, "validation.tfrecords"), no_classes)
    return train_dataset, val_dataset
