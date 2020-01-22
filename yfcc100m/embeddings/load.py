from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tqdm import tqdm
from functools import partial
from common import load_csv_as_dict


class CommaTokenizer(tfds.features.text.Tokenizer):
    def tokenize(self, s):
        """ Splits a string into tokens. As we know our tokens are comma-separated, can just split on comma, which is
            a lot more efficient, as otherwise time complexity is linked to number of word in our vocabulary that
            contain non-alpha numeric characters.
        """
        s = tf.compat.as_text(s)
        toks = s.split(",")
        return toks


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

    classes_encoder = tfds.features.text.TokenTextEncoder(classes_set, decode_token_separator=",",
                                                          tokenizer=CommaTokenizer(alphanum_only=False))
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
    features_encoder = tfds.features.text.TokenTextEncoder(vocab_list, decode_token_separator=",",
                                                           tokenizer=CommaTokenizer(alphanum_only=False))
    return features_encoder


def _str_row_to_tf(proto, features_encoder, classes_encoder, user_tags_limit=None):
    """ Converts comma separated user tags to list of encoded tag id's, and comma separated classes to one hot encoded
        classes

    Parameters
    ----------
    proto : Serialized tf.Tensor
        Serialized row from file produced by yfcc100m.embeddings.prepare.subsets_to_tfrecords
    features_encoder : tfds.features.text.TokenTextEncoder
        User tags encoder
    classes_encoder : tfds.features.text.TokenTextEncoder
        Labels encoder
    user_tags_limit : int
        If an image has more user tags than this value, then the tags beyond this value are ignored. Default of None.
        If None all are kept

    Returns
    -------
    tf.int32, tf.bool

    """

    parsed_features = tf.io.parse_single_example(proto, {
        "FlickrID": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "UserTags": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
        "PredictedConcepts": tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
    })
    user_tags = bytes.decode(parsed_features["UserTags"].numpy()[0])
    labels = bytes.decode(parsed_features["PredictedConcepts"].numpy()[0])
    user_tags = ",".join([tag for tag in user_tags.split(",")][:user_tags_limit])
    labels = ",".join([label_prob.split(":")[0] for label_prob in labels.split(",")])
    encoded_features = [feature for feature in features_encoder.encode(user_tags) if
                        feature != features_encoder.vocab_size - 1]
    encoded_labels = classes_encoder.encode(labels)
    one_hot_labels = np.zeros(classes_encoder.vocab_size)
    for label_num in encoded_labels:
        one_hot_labels[label_num - 1] = 1
    return tf.cast(encoded_features, tf.int32), tf.convert_to_tensor(one_hot_labels, dtype=tf.bool)


def load_subset_as_tf_data(path, classes_encoder, features_encoder, user_tags_limit=None):
    """ Loads the subset passed, encodes the features, and one hot-encodes the classes

    Parameters
    ----------
    path : tf.string
        Path to subset to be loaded
    classes_encoder : tfds.features.text.TokenTextEncoder
        Labels encoder
    features_encoder : tfds.features.text.TokenTextEncoder
        User tags encoder
    user_tags_limit : int
        If an image has more user tags than this value, then the tags beyond this value are ignored. Default of None.
        If None all are kept

    Returns
    -------
    tf.python.data.ops.dataset_ops.DatasetV1Adapter
        The subset ready for use in TensorFlow
    """

    raw_dataset = tf.data.TFRecordDataset(path)
    custom_str_row_to_tf = partial(_str_row_to_tf, user_tags_limit=user_tags_limit, features_encoder=features_encoder,
                                   classes_encoder=classes_encoder)
    features_labels_dataset = raw_dataset.map(
        lambda proto: tf.py_function(custom_str_row_to_tf, inp=[proto], Tout=[tf.int32, tf.bool]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    filtered_features_labels_dataset = features_labels_dataset.filter(
        lambda features, _: tf.not_equal(tf.size(features), 0))
    return filtered_features_labels_dataset


def load_train_val(dataset_folder, classes_encoder, features_encoder, user_tags_limit=None):
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
    user_tags_limit : int
        If an image has more user tags than this value, then the tags beyond this value are ignored. Default of None.
        If None all are kept

    Returns
    -------
    tf.python.data.ops.dataset_ops.DatasetV1Adapter, tf.python.data.ops.dataset_ops.DatasetV1Adapter
        First element is the train data, second is the validation data
    """

    train_dataset = load_subset_as_tf_data(os.path.join(dataset_folder, "train.tfrecords"), classes_encoder,
                                           features_encoder=features_encoder, user_tags_limit=user_tags_limit)
    val_dataset = load_subset_as_tf_data(os.path.join(dataset_folder, "validation.tfrecords"), classes_encoder,
                                         features_encoder=features_encoder, user_tags_limit=user_tags_limit)
    return train_dataset, val_dataset
