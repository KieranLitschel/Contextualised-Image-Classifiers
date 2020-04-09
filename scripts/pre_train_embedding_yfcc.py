import argparse
import math
import os
from functools import partial
from multiprocessing import Process
from time import sleep

import tensorflow as tf
from tensorflow import keras
from tensorflow_core.python.keras.callbacks import ModelCheckpoint, TensorBoard

from embeddings.encoders import CommaTokenTextEncoder
from embeddings.load import build_features_encoder, load_tsv_dataset
from scripts.train_embedding_oiv import evaluate_model


def train(args, config):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with tf.Session():
        tf.set_random_seed(args.random_seed)

        if not os.path.exists("features_encoder.tokens"):
            features_encoder = build_features_encoder(os.path.join(args.oiv_dataset_dir, "train.tsv"),
                                                      tag_threshold=config["tag_threshold"],
                                                      user_tags_limit=config["pad_size"])
        else:
            features_encoder = CommaTokenTextEncoder.load_from_file("features_encoder")
        features_encoder.save_to_file(os.path.join(args.output_dir, "features_encoder"))
        classes_encoder = CommaTokenTextEncoder.load_from_file(args.classes_encoder_path)

        subset_datasets = {}
        subset_samples = {}
        for subset in ["train", "validation"]:
            subset_path = os.path.join(args.yfcc_dataset_dir if subset == "train" else args.oiv_dataset_dir,
                                       "{}.tsv".format(subset))
            subset_samples[subset] = len(open(subset_path, "r").readlines())
            subset_datasets[subset] = load_tsv_dataset(subset_path, features_encoder, classes_encoder,
                                                       config["batch_size"] if subset == "train" else subset_samples[
                                                           subset],
                                                       config["pad_size"], subset_samples[subset],
                                                       True if subset == "train" else False)

        pooling_layer = keras.layers.GlobalAveragePooling1D if config["pooling_layer"] == "GlobalAveragePooling" \
            else keras.layers.GlobalMaxPool1D

        model = keras.Sequential([
            keras.layers.Embedding(features_encoder.vocab_size, config["layer_capacity"]),
            pooling_layer(),
            keras.layers.Dropout(config["dropout_rate"]),
            keras.layers.Dense(config["layer_capacity"], kernel_regularizer=keras.regularizers.l2(), activation='relu'),
            keras.layers.Dropout(config["dropout_rate"]),
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
            epochs=1000000, steps_per_epoch=math.ceil(subset_samples["train"] / config["batch_size"]),
            validation_data=subset_datasets["validation"],
            validation_steps=1, callbacks=[checkpoint, tensorboard],
            verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--yfcc_dataset_dir", help="Location of yfcc dataset produced by build_dataset")
    parser.add_argument("--oiv_dataset_dir", help="Location of oiv dataset produced by build_dataset")
    parser.add_argument("--oiv_human_dataset_dir", help="Location of oiv human dataset produced by build_dataset")
    parser.add_argument("--classes_encoder_path", help="Location of saved encoder produced by build_classes_encoder")
    parser.add_argument("--output_dir", help="Location to save results to")
    parser.add_argument("--random_seed", help="Seed for randomness in experiment", type=int)
    parser.add_argument("--max_train_time", help="Maximum time to train for (in hours)", type=float)

    script_args = parser.parse_args()

    # use best hyperparams found in experiment 1, except we increase the learning rate by a factor of 10 for
    # pre-training
    model_config = {
        "pad_size": 20,
        "tag_threshold": 10,
        "layer_capacity": 64,
        "pooling_layer": "GlobalMaxPool",
        "batch_size": 128,
        "learning_rate": 0.01,
        "dropout_rate": 0.1
    }

    custom_run = partial(train, args=script_args, config=model_config)

    p = Process(target=custom_run)

    p.daemon = True

    p.start()

    run_time_limit = 60 * 60 * script_args.max_train_time
    sleep(run_time_limit)

    p.terminate()

    os.system(r"kill -9 $(nvidia-smi | sed -n 's/|\s*[0-9]*\s*\([0-9]*\)\s*.*/\1/p' | sort | uniq | sed '/^$/d')")

    sleep(60)

    print("Finished training, evaluating best model")

    evaluate_model(script_args.output_dir, script_args.classes_encoder_path, script_args.oiv_dataset_dir,
                   script_args.oiv_human_dataset_dir, model_config["pad_size"])
