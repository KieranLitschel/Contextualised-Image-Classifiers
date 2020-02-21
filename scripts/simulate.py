import os
import urllib
import argparse

import tensorflow as tf

from embeddings.prepare import pre_process_user_tags
from embeddings.encoders import CommaTokenTextEncoder
import numpy as np
from oiv.common import get_oiv_labels_to_human, get_class_descriptions_path

parser = argparse.ArgumentParser()

parser.add_argument("--pad_size", help="Maximum number of words per image", type=int)
parser.add_argument("--classes_encoder_path", help="Location of saved encoder produced by build_classes_encoder",
                    type=str)
parser.add_argument("--input_dir", help="Location of model and feature encoder", type=str)
parser.add_argument("--top_n", help="Top n predictions to return", type=int)

args = parser.parse_args()

with tf.Session():
    best_model = tf.keras.models.load_model(os.path.join(args.input_dir, 'best_model.h5'))
    features_encoder = CommaTokenTextEncoder.load_from_file(os.path.join(args.input_dir, "features_encoder"))
    classes_encoder = CommaTokenTextEncoder.load_from_file(args.classes_encoder_path)
    oiv_labels_to_human = get_oiv_labels_to_human(get_class_descriptions_path())
    while True:
        words = input("Enter at max {} comma-separated tags: ".format(args.pad_size))
        processed_words = ",".join(
            ["+".join([urllib.parse.quote(token.lower()) for token in word.split(" ")]) for word in
             words.split(",")[:args.pad_size]])
        pre_processed_words = pre_process_user_tags(processed_words, remove_nums=True, stem=False)
        encoded_words = features_encoder.encode(pre_processed_words)
        for _ in range(len(encoded_words), args.pad_size):
            encoded_words.append(0)
        encoded_words = np.array([encoded_words])
        predictions = best_model.predict(encoded_words)
        top_n_pred = [(index + 1, score) for index, score in
                      sorted(enumerate(list(predictions[0])), reverse=True, key=lambda x: x[1])[:args.top_n]]
        top_n_classes = [oiv_labels_to_human[label] for label in
                         classes_encoder.decode([class_num for class_num, score in top_n_pred]).split(",")]
        for i in range(args.top_n):
            print("{} with score {}".format(top_n_classes[i], top_n_pred[i][1]))
        print("-----------------------------------------")
