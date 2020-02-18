from __future__ import absolute_import, division, print_function, unicode_literals

from functools import partial

import numpy as np
import tensorflow as tf
from tqdm import tqdm

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
        if count >= tag_threshold:
            vocab_list.append(vocab)
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


def pre_process_tsv_row(user_tags, label_confidences, features_encoder, classes_encoder, pad_size):
    """ Pre-processes a samples user_tags and labels_confidences

    Parameters
    ----------
    user_tags : tf.Tensor
        Tensor containing single encoded string consisting of comma-separated user tags
    label_confidences : tf.Tensor
        Tensor containing single encoded string consisting of comma-separated labels, with a confidence value for each
    features_encoder : embeddings.encoders.CommaTokenTextEncoder
        Encoder for words
    classes_encoder : embeddings.encoders.CommaTokenTextEncoder
        Encoder for classes
    pad_size : int
        Amount to pad user_tags to, if there are more user tags than this value, then all after pad_size are ignored

    Returns
    -------
    tf.Tensor, tf.Tensor
        Tensor containing encoded user tags for the sample, and another containing a serialized SparseTensor describing
        the confidence for each label
    """

    user_tags = user_tags.numpy()[0].decode()
    encoded_user_tags = np.array(features_encoder.encode(user_tags), dtype=np.int32)[:pad_size]
    label_confidences = label_confidences.numpy()[0].decode()
    if label_confidences:
        label_confidences = [label_confidence.split(":")
                             for label_confidence in label_confidences.split(",")]
        confidence = [float(confidence) for _, confidence in label_confidences]
        # encoder indexes labels 1 to 500, we subtract 0 so they are indexed 0 to 499
        encoded_labels = [label - 1 for label in
                          classes_encoder.encode(",".join([label for label, _ in label_confidences]))]
        confidences = np.array([confidence for _, confidence in sorted(zip(encoded_labels, confidence))],
                               dtype=np.float32)
        encoded_labels = np.array([[0, label] for label in sorted(encoded_labels)], dtype=np.int32)
    else:
        # if no labels then make the probability of all labels 0
        encoded_labels = np.empty((0, 2), dtype=np.int32)
        confidences = np.array([], dtype=np.float32)
    sparse_labels = tf.SparseTensor(indices=encoded_labels, values=confidences,
                                    dense_shape=[1, classes_encoder.vocab_size - 2])
    return tf.cast(encoded_user_tags, dtype=tf.int32), tf.cast(tf.serialize_sparse(sparse_labels), dtype=tf.string)


def sparse_labels_to_dense(encoded_user_tags, sparse_labels):
    """ Expands the sparse serialized labels to a dense Tensor

    Parameters
    ----------
    encoded_user_tags : tf.Tensor
        Tensor containing encoded and padded user tags for the sample
    sparse_labels : tf.Tensor
        Tensor containing a serialized SparseTensor describing the confidence for each label

    Returns
    -------
    tf.Tensor, tf.Tensor
        Unchanged user tags, and sparse_labels deserialized and converted to a dense Tensor
    """

    dense_labels = tf.sparse.to_dense(tf.deserialize_many_sparse(sparse_labels, dtype=tf.float32))
    dense_labels = tf.reshape(dense_labels, [dense_labels.shape[0], -1])
    return encoded_user_tags, dense_labels


def sparse_labels_to_dense_wrapper(encoded_user_tags, sparse_labels, pad_size, batch_size, no_classes):
    """ Wraps sparse_labels_to_dense, asserting their shape, to fix issue discussed here
        https://github.com/tensorflow/tensorflow/issues/24520

    Parameters
    ----------
    encoded_user_tags : tf.Tensor
        Tensor containing encoded and padded user tags for the sample
    sparse_labels : tf.Tensor
        Tensor containing a serialized SparseTensor describing the confidence for each label
    pad_size : int
        Amount to pad user_tags to, if there are more user tags than this value, then all after pad_size are ignored
    batch_size : int
        Size of batches
    no_classes : int
        Number of classes

    Returns
    -------
    tf.Tensor, tf.Tensor
        Unchanged user tags, and sparse_labels deserialized and converted to a dense Tensor
    """

    encoded_user_tags, dense_labels = tf.py_function(sparse_labels_to_dense,
                                                     [encoded_user_tags, sparse_labels],
                                                     Tout=[tf.int32, tf.float32])
    encoded_user_tags.set_shape((batch_size, pad_size))
    dense_labels.set_shape((batch_size, no_classes))
    return encoded_user_tags, dense_labels


def load_tsv_dataset(dataset_path, features_encoder, classes_encoder, batch_size, pad_size, no_samples, shuffle):
    """ Loads the subset tsv, preparing it for training

    Parameters
    ----------
    dataset_path : str
        Path to subset to be loaded
    features_encoder : embeddings.encoders.CommaTokenTextEncoder
        Encoder for words
    classes_encoder : embeddings.encoders.CommaTokenTextEncoder
        Encoder for classes
    batch_size : int
        Size of batches
    pad_size : int
        Amount to pad user_tags to, if there are more user tags than this value, then all after pad_size are ignored
    no_samples : int
        Number of samples in the subset
    shuffle : bool
        Whether to shuffle the dataset each iteration before sampling

    Returns
    -------
    tf.python.data.ops.dataset_ops.DatasetV1Adapter
        The pre-processed, cached, shuffled, padded, prefetched dataset
    """

    custom_pre_process_tsv_row = partial(pre_process_tsv_row, features_encoder=features_encoder,
                                         classes_encoder=classes_encoder, pad_size=pad_size)
    custom_sparse_labels_to_dense_wrapper = partial(sparse_labels_to_dense_wrapper, pad_size=pad_size,
                                                    batch_size=batch_size, no_classes=classes_encoder.vocab_size - 2)
    dataset = tf.data.experimental.make_csv_dataset(dataset_path,
                                                    column_names=["flickr_id", "user_tags", "labels"],
                                                    label_name="labels", select_columns=["user_tags", "labels"],
                                                    field_delim="\t", header=False, shuffle=False, batch_size=1) \
        .map(lambda features, label_confidences: tf.py_function(custom_pre_process_tsv_row,
                                                                [features["user_tags"], label_confidences],
                                                                Tout=[tf.int32, tf.string]),
             num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .take(no_samples) \
        .cache()
    if shuffle:
        dataset = dataset.shuffle(no_samples, reshuffle_each_iteration=True)
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None], [None])) \
        .map(custom_sparse_labels_to_dense_wrapper) \
        .prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
