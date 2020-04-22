import os
import pathlib
import urllib

import tensorflow as tf

from yfcc100m.dataset import pre_process_user_tags
from embeddings.encoders import CommaTokenTextEncoder
import numpy as np
from oiv.common import get_oiv_labels_to_human, get_class_descriptions_path

proj_root = pathlib.Path(__file__).parent.parent.absolute()

pad_size = 20
top_n = 5
input_dir = os.path.join(proj_root, "experiment_results/basic/repeat 2")
classes_encoder_path = os.path.join(proj_root, "experiment_results/classes_encoder")

with tf.Session():
    best_model = tf.keras.models.load_model(os.path.join(input_dir, 'best_model.h5'))
    features_encoder = CommaTokenTextEncoder.load_from_file(os.path.join(input_dir, "features_encoder"))
    classes_encoder = CommaTokenTextEncoder.load_from_file(classes_encoder_path)
    oiv_labels_to_human = get_oiv_labels_to_human(get_class_descriptions_path())
    while True:
        words = input("Enter at max {} comma-separated tags: ".format(pad_size))
        processed_words = ",".join(
            ["+".join([urllib.parse.quote(token.lower()) for token in word.split(" ")]) for word in
             words.split(",")[:pad_size]])
        pre_processed_words = pre_process_user_tags(processed_words, remove_nums=True, stem=False)
        encoded_words = features_encoder.encode(pre_processed_words)
        for _ in range(len(encoded_words), pad_size):
            encoded_words.append(0)
        encoded_words = np.array([encoded_words])
        predictions = best_model.predict(encoded_words)
        top_n_pred = [(index + 1, score) for index, score in
                      sorted(enumerate(list(predictions[0])), reverse=True, key=lambda x: x[1])[:top_n]]
        top_n_classes = [oiv_labels_to_human[label] for label in
                         classes_encoder.decode([class_num for class_num, score in top_n_pred]).split(" ")]
        for i in range(top_n):
            print("{} with confidence {}".format(top_n_classes[i], top_n_pred[i][1]))
        print("-----------------------------------------")
