from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tqdm import tqdm
from functools import partial


def _build_classes_encoder(classes_set):
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


def _process_raw_row(row):
    """ Takes a set of classes and builds a token text encoder, to convert the str classes to numbers

    Parameters
    ----------
    row : tf.string
        Row from file produced by yfcc100m.embeddings.prepare.joined_to_subsets

    Returns
    -------
    (str, str)
        First element is comma separated user tags, second element is comma separated classes, both in str format
    """

    _, user_tags, labels = bytes.decode(row.numpy()).split("\t")
    labels = ",".join([label_prob.split(":")[0] for label_prob in labels.split(",")])
    return tf.cast(user_tags, tf.string), tf.cast(labels, tf.string)


def _str_row_to_int(features, labels, features_encoder, classes_encoder):
    """ Converts comma separated user tags to list of encoded tag id's, and comma separated classes to one hot encoded
        classes

    Parameters
    ----------
    features : tf.string
        User tags, separated by commas
    labels : tf.string
        Labels, separated by commas
    features_encoder : tfds.features.text.TokenTextEncoder
        User tags encoder
    classes_encoder : tfds.features.text.TokenTextEncoder
        Labels encoder

    Returns
    -------
    (np array of int, np array of int)
        First element is
    """

    encoded_features = features_encoder.encode(bytes.decode(features.numpy()))
    encoded_labels = classes_encoder.encode(bytes.decode(labels.numpy()))
    one_hot_labels = np.zeros(classes_encoder.vocab_size)
    for label_num in encoded_labels:
        one_hot_labels[label_num - 1] = 1
    return encoded_features, one_hot_labels


def load_subset_as_tf_data(path, classes_encoder, features_encoder=None, tag_threshold=None):
    """ Loads the subset passed, encodes the features, and one hot-encodes the classes

    Parameters
    ----------
    path : tf.string
        Path to subset to be loaded
    classes_encoder : tfds.features.text.TokenTextEncoder
        Labels encoder
    features_encoder : tfds.features.text.TokenTextEncoder
        User tags encoder. If not specified build from scratch
    tag_threshold : int
        Used to construct feature extractor if it is not specified. Threshold for keeping tags as features. Default of
        None. If None all tags are kept

    Returns
    -------
    tf.python.data.ops.dataset_ops.DatasetV1Adapter
        The subset ready for use in TensorFlow, if feature_encoder was not passed then the constructed one is returned
        as the second element of a tuple
    """

    raw_dataset = tf.data.TextLineDataset(path)
    str_features_labels_dataset = raw_dataset.map(
        lambda row: tf.py_function(_process_raw_row, inp=[row], Tout=[tf.string, tf.string]))
    feature_encoder_was_none = not features_encoder
    if not features_encoder:
        print("Building feature encoder")
        vocab_count = {}
        for user_tags, _ in tqdm(str_features_labels_dataset):
            tokens = bytes.decode(user_tags.numpy()).split(",")
            for token in tokens:
                if not vocab_count.get(token):
                    vocab_count[token] = 0
                vocab_count[token] += 1
        vocab_list = []
        for vocab, count in vocab_count.items():
            if not tag_threshold or count > tag_threshold:
                vocab_list.append(vocab)
        features_encoder = tfds.features.text.TokenTextEncoder(vocab_list, decode_token_separator=",")
    custom_str_row_to_int = partial(_str_row_to_int, features_encoder=features_encoder, classes_encoder=classes_encoder)
    features_labels_dataset = str_features_labels_dataset.map(
        lambda features, labels: tf.py_function(custom_str_row_to_int, inp=[features, labels],
                                                Tout=[tf.int64, tf.int64]))
    if feature_encoder_was_none:
        return features_labels_dataset, features_encoder
    return features_labels_dataset


def load_train_val(dataset_folder, classes_set, tag_threshold=None):
    """ For train and validation, loads them, encodes the features, and one hot-encodes the classes. Validation dataset
        features encoded using encoder built from train

    Parameters
    ----------
    dataset_folder : str
        The location of the train and validation files
    classes_set : set of str
        The set of classes
    tag_threshold : int
        Threshold for keeping tags as features. Default of None. If None all tags are kept

    Returns
    -------
    tf.python.data.ops.dataset_ops.DatasetV1Adapter, tf.python.data.ops.dataset_ops.DatasetV1Adapter,
    tfds.features.text.TokenTextEncoder
        First element is the train data, second is the validation data, third is the feature extractor built from train
    """

    classes_encoder = _build_classes_encoder(classes_set)
    train_dataset, features_encoder = load_subset_as_tf_data(os.path.join(dataset_folder, "train"), classes_encoder,
                                                             tag_threshold=tag_threshold)
    val_dataset = load_subset_as_tf_data(os.path.join(dataset_folder, "validation"), classes_encoder,
                                         features_encoder=features_encoder)
    return train_dataset, val_dataset, features_encoder
