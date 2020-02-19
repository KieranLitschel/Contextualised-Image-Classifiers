import argparse
import math
import os

from tensorflow import keras
from tensorflow_core.python.keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow as tf

from embeddings.encoders import CommaTokenTextEncoder
from embeddings.load import build_features_encoder, load_tsv_dataset
from oiv.evaluate import build_oid_challenge_image_level_map_func
from oiv.common import get_class_descriptions_path

parser = argparse.ArgumentParser()

parser.add_argument("--pad_size", help="Amount to pad feature vector by so they all have same number of user tags. If "
                                       "images have more user tags than pad size, user tags beyond pad size are "
                                       "ignored.", type=int)
parser.add_argument("--tag_threshold", help="Threshold required for a tag to be a feature, otherwise will be "
                                            "considered unknown", type=int)
parser.add_argument("--layer_capacity", help="Number of units per layer", type=int)
parser.add_argument("--global_average_pooling",
                    help="Whether to use GlobalAveragePooling, if False uses GlobalMaxPooling", type=bool)
parser.add_argument("--oiv_dataset_dir", help="Location of oiv dataset produced by build_dataset")
parser.add_argument("--oiv_human_dataset_dir", help="Location of oiv human dataset produced by build_dataset")
parser.add_argument("--classes_encoder_path", help="Location of saved encoder produced by build_classes_encoder")
parser.add_argument("--output_dir", help="Location to save results to")
parser.add_argument("--random_seed", help="Seed for randomness in experiment", type=int)
parser.add_argument("--batch_size", help="Size for each batch", type=int)
parser.add_argument("--epochs", help="Number of epochs to train for", type=int)
parser.add_argument("--dropout_rate", help="Rate of dropout", type=float)

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

with tf.Session() as sess:
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

    human_validation_path = os.path.join(args.oiv_human_dataset_dir, "validation.tsv")
    label_names_file = get_class_descriptions_path()

    oid_challenge_image_level_map = build_oid_challenge_image_level_map_func(human_validation_path, label_names_file,
                                                                             classes_encoder)

    pooling_layer = keras.layers.GlobalAveragePooling1D if args.global_average_pooling else keras.layers.GlobalMaxPool1D

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
                  loss='binary_crossentropy', metrics=[oid_challenge_image_level_map])

    checkpoint = ModelCheckpoint(os.path.join(args.output_dir, 'model.h5'), verbose=1,
                                 monitor='val_oid_challenge_image_level_map',
                                 save_best_only=True, mode='auto')

    log_dir = os.path.join(args.output_dir, "log")
    tensorboard = TensorBoard(log_dir)

    history = model.fit(
        subset_datasets["train"],
        epochs=args.epochs, steps_per_epoch=math.ceil((subset_samples["train"] / args.batch_size) / 2),
        validation_data=subset_datasets["validation"],
        validation_steps=1, callbacks=[checkpoint, tensorboard],
        verbose=True)
