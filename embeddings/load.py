from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
from tqdm import tqdm
from functools import partial
from common import load_csv_as_dict
from embeddings.encoders import CommaTokenTextEncoder


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


def build_features_encoder(subset_path):
    """ Takes a set of classes and builds a token text encoder, to convert the str classes to numbers

    Parameters
    ----------
    subset_path : str
        Path to subset to build encoder from

    Returns
    -------
    CommaTokenTextEncoder
        Encoder for features
    """

    vocab_count = count_user_tags(subset_path)
    vocab_list = vocab_count.keys()
    features_encoder = CommaTokenTextEncoder(vocab_list, decode_token_separator=",")
    return features_encoder


def _batch_decode_pad_one_hot(batch):
    """ Decodes batches, padding with 0 such that all samples in the batch have the same number of features, and one
        hot encodes the labels

    Parameters
    ----------
    batch : Serialized tf.Tensor
        Serialized batch from file produced by build_dataset

    Returns
    -------
    tf.int32, tf.bool
        First element is features, second element is one hot labels
    """

    """
    parsed_batch = tf.io.parse_example(batch, {
        "flickr_id": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "encoded_features": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "encoded_labels": tf.io.FixedLenSequenceFeature([], tf.str, allow_missing=True),
    })
    encoded_features = tf.cast(parsed_batch["encoded_features"].numpy(), dtype=tf.int32)
    encoded_labels = tf.stack([tf.sparse_tensor_to_dense(tf.parse_tensor(serial_tensor)) for serial_tensor in
                               parsed_batch["encoded_labels"].numpy()])
    return tf.cast(encoded_features, dtype=tf.int32), tf.convert_to_tensor(encoded_labels, dtype=tf.float32)
    """
    raise NotImplemented


def load_subset_as_tf_data(path, no_classes, batch_size):
    """ Loads the subset passed, encodes the features, and one hot-encodes the classes

    Parameters
    ----------
    path : tf.string
        Path to subset to be loaded
    no_classes : int
        Number of classes
    batch_size : int
        Size of batches to be loaded

    Returns
    -------
    tf.python.data.ops.dataset_ops.DatasetV1Adapter
        The subset ready for use in TensorFlow
    """

    """
    custom_row_to_tf = partial(_batch_decode_pad_one_hot, no_classes=no_classes)
    dataset = tf.data.TFRecordDataset(path) \
        .shuffle(batch_size * 2) \
        .batch(batch_size) \
        .map(lambda batch: tf.py_function(custom_row_to_tf, [batch], Tout=[tf.int32, tf.float32]))
    return dataset
    """
    raise NotImplemented


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

    """
    train_dataset = load_subset_as_tf_data(os.path.join(dataset_folder, "train.tfrecords"), no_classes)
    val_dataset = load_subset_as_tf_data(os.path.join(dataset_folder, "validation.tfrecords"), no_classes)
    return train_dataset, val_dataset
    """
    raise NotImplemented
