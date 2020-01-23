import multiprocessing

import tensorflow as tf
import os

from yfcc100m.embeddings.prepare import build_dataset


def build(tag_threshold, user_tags_limit):
    with tf.Session():
        root_folder = "/dev/shm/YFCC100M"
        dataset_folder = os.path.join(root_folder, "dataset")
        output_folder = os.path.join(dataset_folder,
                                     "tag_thresh_{}_user_tags_limit_{}".format(tag_threshold, user_tags_limit))
        classes_encoder_path = os.path.join(root_folder, "yfcc100m_classes_encoder")

        build_dataset(dataset_folder, classes_encoder_path, output_folder, tag_threshold,
                      user_tags_limit)


if __name__ == "__main__":
    hyper_params = [(tag_thresh, user_tag_lim) for tag_thresh in [100, 1000] for user_tag_lim in [2, 5, 10]]

    pool = multiprocessing.Pool(6)
    pool.starmap(build, hyper_params)
