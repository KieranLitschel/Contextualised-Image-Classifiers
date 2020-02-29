import argparse
import logging
import math
import os
from functools import partial
from multiprocessing import Process
from time import sleep

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow_core.python.keras.callbacks import ModelCheckpoint, TensorBoard

from embeddings.encoders import CommaTokenTextEncoder
from embeddings.load import build_features_encoder, load_tsv_dataset
from oiv.common import get_class_descriptions_path
from oiv.evaluate import build_y_true, build_categories, oid_challenge_evaluator_image_level


def train(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with tf.Session():
        tf.set_random_seed(args.random_seed)

        if not os.path.exists("features_encoder.tokens"):
            features_encoder = build_features_encoder(os.path.join(args.oiv_dataset_dir, "train.tsv"),
                                                      tag_threshold=args.tag_threshold,
                                                      user_tags_limit=args.pad_size)
        else:
            features_encoder = CommaTokenTextEncoder.load_from_file("features_encoder")
        features_encoder.save_to_file(os.path.join(args.output_dir, "features_encoder"))
        classes_encoder = CommaTokenTextEncoder.load_from_file(args.classes_encoder_path)

        subset_datasets = {}
        subset_samples = {}
        for subset in ["train", "validation"]:
            subset_path = os.path.join(args.oiv_dataset_dir, "{}.tsv".format(subset))
            subset_samples[subset] = len(open(subset_path, "r").readlines())
            subset_datasets[subset] = load_tsv_dataset(subset_path, features_encoder, classes_encoder,
                                                       args.batch_size if subset == "train" else subset_samples[subset],
                                                       args.pad_size, subset_samples[subset],
                                                       True if subset == "train" else False)

        pooling_layer = keras.layers.GlobalAveragePooling1D if args.pooling_layer == "GlobalAveragePooling" \
            else keras.layers.GlobalMaxPool1D

        model = keras.Sequential([
            keras.layers.Embedding(features_encoder.vocab_size, args.layer_capacity),
            pooling_layer(),
            keras.layers.Dropout(args.dropout_rate),
            keras.layers.Dense(args.layer_capacity, kernel_regularizer=keras.regularizers.l2(), activation='relu'),
            keras.layers.Dropout(args.dropout_rate),
            keras.layers.Dense(classes_encoder.vocab_size - 2, activation='sigmoid')
        ])

        model.summary()

        model.compile(optimizer='adam',
                      loss='binary_crossentropy')

        checkpoint = ModelCheckpoint(os.path.join(args.output_dir, 'best_model.h5'), verbose=1,
                                     monitor='val_loss',
                                     save_best_only=True, mode='auto')

        log_dir = os.path.join(args.output_dir, "log")
        tensorboard = TensorBoard(log_dir)

        model.fit(
            subset_datasets["train"],
            epochs=5000, steps_per_epoch=math.ceil(subset_samples["train"] / args.batch_size),
            validation_data=subset_datasets["validation"],
            validation_steps=1, callbacks=[checkpoint, tensorboard],
            verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pad_size",
                        help="Amount to pad feature vector by so they all have same number of user tags. If "
                             "images have more user tags than pad size, user tags beyond pad size are "
                             "ignored.", type=int)
    parser.add_argument("--tag_threshold", help="Threshold required for a tag to be a feature, otherwise will be "
                                                "considered unknown", type=int)
    parser.add_argument("--layer_capacity", help="Number of units per layer", type=int)
    parser.add_argument("--pooling_layer",
                        help="Pooling layer to use GlobalAveragePooling or GlobalMaxPooling", type=str)
    parser.add_argument("--batch_size", help="Size for each batch", type=int)
    parser.add_argument("--learning_rate", help="Learning rate", type=float)
    parser.add_argument("--dropout_rate", help="Rate of dropout", type=float)
    parser.add_argument("--l2_reg_factor", help="Regularization factor for L2 regularization", type=float)
    parser.add_argument("--oiv_dataset_dir", help="Location of oiv dataset produced by build_dataset")
    parser.add_argument("--oiv_human_dataset_dir", help="Location of oiv human dataset produced by build_dataset")
    parser.add_argument("--classes_encoder_path", help="Location of saved encoder produced by build_classes_encoder")
    parser.add_argument("--output_dir", help="Location to save results to")
    parser.add_argument("--random_seed", help="Seed for randomness in experiment", type=int)
    parser.add_argument("--max_train_time", help="Maximum time to train for (in hours)", type=float)

    script_args = parser.parse_args()

    custom_run = partial(train, args=script_args)

    p = Process(target=custom_run)

    p.daemon = True

    p.start()

    run_time_limit = 60 * 60 * script_args.max_train_time
    sleep(run_time_limit)

    p.terminate()

    os.system(r"kill -9 $(nvidia-smi | sed -n 's/|\s*[0-9]*\s*\([0-9]*\)\s*.*/\1/p' | sort | uniq | sed '/^$/d')")

    sleep(60)

    print("Finished training, evaluating best model")

    with tf.Session():
        best_model = tf.keras.models.load_model(os.path.join(script_args.output_dir, 'best_model.h5'))
        best_model.summary()

        eval_classes_encoder = CommaTokenTextEncoder.load_from_file(script_args.classes_encoder_path)
        eval_features_encoder = CommaTokenTextEncoder.load_from_file(
            os.path.join(script_args.output_dir, "features_encoder"))

        validation_path = os.path.join(script_args.oiv_dataset_dir, "validation.tsv")
        validation_samples = len(open(validation_path, "r").readlines())
        validation_dataset = load_tsv_dataset(validation_path, eval_features_encoder, eval_classes_encoder,
                                              validation_samples, script_args.pad_size, validation_samples, False)

        y_pred = best_model.predict(validation_dataset, steps=1)
        y_true = build_y_true(os.path.join(script_args.oiv_human_dataset_dir, "validation.tsv"), eval_classes_encoder)
        categories = build_categories(get_class_descriptions_path(), eval_classes_encoder)

        # the challenge evaluator will throw warnings about classes being missing, and the
        # groundtruth group_of flag being missing, but we don't care about them and they clutter
        # the command line, so we ignore them
        logger = logging.getLogger()
        logger.disabled = True
        metrics = oid_challenge_evaluator_image_level(y_pred, y_true, categories)
        logger.disabled = False

        df_metrics = pd.DataFrame(list(metrics.items()))
        df_metrics.to_csv(os.path.join(script_args.output_dir, "best_model_validation_metrics.csv"))

        print("Validation MAP is: {}".format(metrics["OpenImagesDetectionChallenge_Precision/mAP@0.5IOU"]))
