import os

import numpy as np
from object_detection.core import standard_fields
from research.object_detection.utils.object_detection_evaluation import OpenImagesDetectionChallengeEvaluator

from common import load_csv_as_dict
from oiv.common import get_oiv_labels_to_human, get_image_id_flickr_id_map
import tensorflow as tf
import warnings

def build_categories(label_names_file, classes_encoder):
    """ Builds the categories dictionary for OpenImagesDetectionChallengeEvaluator

    Parameters
    ----------
    label_names_file : str
        Path to file mapping OIV labels to OIV names
    classes_encoder : embeddings.encoders.CommaTokenTextEncoder
        Encoder for classes

    Returns
    -------
    list of dict
        A list of dicts, each of which has the following keys -
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name e.g., 'cat', 'dog'.

    """

    oiv_label_human_map = get_oiv_labels_to_human(label_names_file)
    labels = classes_encoder.tokens
    categories = []
    for label in labels:
        # don't need to subtract 1 from class encoders ids here, as category id's are indexed from 1
        category = {"id": classes_encoder.encode(label)[0], "name": oiv_label_human_map[label]}
        categories.append(category)
    return categories


def build_y_true(subset_path, classes_encoder):
    """ Builds the ground truth labels into a data structure that we can iterate over to add groundtruths to
        OpenImagesDetectionChallengeEvaluator

    Parameters
    ----------
    subset_path : str
        Path to oiv_human_verified subset built by join_dataset_and_autotags
    classes_encoder : embeddings.encoders.CommaTokenTextEncoder
        Encoder for classes

    Returns
    -------
    list of dict of str -> list of int
        A list where each element is a dictionary corresponding to the respective image in the subset. The key "Pos"
        contains a list of classes identified as being in the image, represented by their 1 indexed class encodings.
        The key "Pres" contains a list of classes that have been identified as either being or not being in the image
        (so their ground truth is not unknown), represented by their 1 indexed class encodings

    """

    subset = load_csv_as_dict(subset_path, fieldnames=["flickr_id", "user_tags", "labels"], delimiter="\t")
    y_trues = []
    for row in subset:
        labels = row["labels"]
        label_confidences = [label_confidence.split(":")
                             for label_confidence in labels.split(",")]
        pres = []
        pos = []
        for label, confidence in label_confidences:
            confidence = float(confidence)
            # don't need to subtract 1 from class encoders ids here, as category id's are indexed from 1
            encoded_label = classes_encoder.encode(label)[0]
            if confidence == 1:
                pos.append(encoded_label)
            pres.append(encoded_label)
        if pos:
            pos = sorted(pos)
        if pres:
            pres = sorted(pres)
        y_true = {"Pos": pos, "Pres": pres}
        y_trues.append(y_true)
    return y_trues


def oid_challenge_evaluator_image_level(y_pred, y_true, categories):
    """ Evaluate the image level predictions using OpenImagesDetectionChallengeEvaluator, by saying the ground truths
        correspond to a box the size of the entire image, and predictions correspond to the same whole image sized box

    Parameters
    ----------
    y_pred : np.array
        Numpy array of shape (no. of images, no. of classes), describing the predicted probability that each class
        belongs to each image
    y_true : list of dict of str -> np.array
        List of dicts of numpy arrays, where the ith tuple corresponds to the ith image in the validation set.
        Value of key "Pos" corresponds to a numpy array listing all classes which are present in the image. Value
        of key "Pres" corresponds to a numpy array listing all classes which have been verified (as present or not
        present) in the image
    categories : list of dict
        A list of dicts, each of which has the following keys - 'id': (required) an integer id uniquely identifying this
        category. 'name': (required) string representing category name e.g., 'cat', 'dog'.

    Returns
    -------
    dict of str -> float
        A dictionary of metrics with the following fields -

        1. summary_metrics:
           '<prefix if not empty>_Precision/mAP@<matching_iou_threshold>IOU': mean
           average precision at the specified IOU threshold.
        2. per_category_ap: category specific results with keys of the form
           <prefix if not empty>_PerformanceByCategory/
           mAP@<matching_iou_threshold>IOU/category'.

    """

    evaluator = OpenImagesDetectionChallengeEvaluator(categories)
    for image_id, image_details in enumerate(y_true):
        pres = np.array(image_details["Pres"])
        if image_details["Pos"]:
            pos = np.array(image_details["Pos"])
            boxes = np.array(
                [np.array([0, 0, 1, 1], dtype=np.float32) for _ in range(pos.shape[0])])
        else:
            # this seems to be the way to handle if an image is known only not to contain some classes
            pos = np.array([0])
            boxes = np.array([[0, 0, 1, 1]], dtype=np.float32)
        groundtruth_dict = {
            standard_fields.InputDataFields.groundtruth_boxes: boxes,
            standard_fields.InputDataFields.groundtruth_classes: pos,
            standard_fields.InputDataFields.groundtruth_image_classes: pres,
        }
        evaluator.add_single_ground_truth_image_info(image_id, groundtruth_dict)
    sorted_image_ids = np.array(sorted(cat_dict["id"] for cat_dict in categories))
    boxes = np.array([np.array([0, 0, 1, 1], dtype=np.float32) for _ in range(sorted_image_ids.shape[0])])
    for image_id, image_preds in enumerate(y_pred):
        detection_dict = {
            standard_fields.DetectionResultFields.detection_boxes: boxes,
            standard_fields.DetectionResultFields.detection_scores: image_preds,
            standard_fields.DetectionResultFields.detection_classes: sorted_image_ids,
        }
        evaluator.add_single_detected_image_info(image_id, detection_dict)
    metrics = evaluator.evaluate()
    return metrics


def build_oid_challenge_image_level_map_func(human_verified_subset_path, label_names_file, classes_encoder):
    """ Creates a function that takes y_pred and y_true (human and machine generated labels), and finds the mean average
        precision for y_pred using y_true_human (only human generated machine labels)

    Parameters
    ----------
    human_verified_subset_path : str
        Path to oiv_human_verified subset built by join_dataset_and_autotags
    label_names_file : str
        Path to file mapping OIV labels to OIV names
    classes_encoder : embeddings.encoders.CommaTokenTextEncoder
        Encoder for classes

    """

    categories = build_categories(label_names_file, classes_encoder)
    y_true_human = build_y_true(human_verified_subset_path, classes_encoder)

    def oid_challenge_image_level_map(y_pred, _):
        def setup_and_evaluate(y_pred_):
            y_pred_other = y_pred_.numpy().astype(np.float32)
            # Keras will call this function for the training set, but we don't want to evaluate it for it
            # so we can check the length of y_pred_, and if it's our validation set, we can evaluate it,
            # otherwise we can just say the result is 0
            result = 0
            if y_pred_other.shape[0] == len(y_true_human):
                with warnings.catch_warnings():
                    # the challenge evaluator will throw warnings about classes being missing, and the
                    # groundtruth group_of flag being missing, but we don't care about them and they clutter
                    # the command line, so we ignore them
                    warnings.simplefilter("ignore")
                    result = oid_challenge_evaluator_image_level(y_pred_other, y_true_human,
                                                                 categories)["OpenImagesDetectionChallenge_"
                                                                             "Precision/mAP@0.5IOU"]
            return tf.convert_to_tensor(result, dtype=tf.float32)

        mean_ap = tf.py_function(setup_and_evaluate, [y_pred], Tout=tf.float32)
        return mean_ap

    return oid_challenge_image_level_map


def machine_labels_baseline(subset, oiv_folder, oiv_human_verified_folder, label_names_file, classes_encoder):
    """ FOO BAR

    Parameters
    ----------
    subset : str
        Subset to evaluate the machine labels for. Should be one of "train", "validation", or "test"
    oiv_folder : str
        Path to folder containing open images csv's
    oiv_human_verified_folder : str
        Path to oiv_human_verified folder built by join_dataset_and_autotags
    label_names_file : str
        Path to file mapping OIV labels to OIV names
    classes_encoder : embeddings.encoders.CommaTokenTextEncoder
        Encoder for classes

    Returns
    -------
    dict of str -> float
        A dictionary of metrics with the following fields -

        1. summary_metrics:
           '<prefix if not empty>_Precision/mAP@<matching_iou_threshold>IOU': mean
           average precision at the specified IOU threshold.
        2. per_category_ap: category specific results with keys of the form
           <prefix if not empty>_PerformanceByCategory/
           mAP@<matching_iou_threshold>IOU/category'.

    """
    machine_labels_path = os.path.join(oiv_folder, "{}-annotations-machine-imagelabels.csv".format(subset))
    human_verified_subset_path = os.path.join(oiv_human_verified_folder, "{}.tsv".format(subset))
    rotations_path = os.path.join(oiv_folder, "{}-images-with-rotation.csv".format(subset))
    human_verified_path = os.path.join(oiv_human_verified_folder, "{}.tsv".format(subset))
    subset_flickr_ids = [row["flickr_id"] for row in
                         load_csv_as_dict(human_verified_path, fieldnames=["flickr_id", "user_tags", "labels"],
                                          delimiter="\t")]
    subset_flickr_ids_set = set(subset_flickr_ids)
    image_id_flickr_id_map = get_image_id_flickr_id_map(rotations_path)
    machine_labels_csv = load_csv_as_dict(machine_labels_path, delimiter=",")
    image_label_confidences = {}
    for row in machine_labels_csv:
        image_id = row["ImageID"]
        # there are some labels for images that aren't in the validation set
        if not image_id_flickr_id_map.get(image_id):
            continue
        flickr_id = image_id_flickr_id_map[image_id]
        if flickr_id not in subset_flickr_ids_set:
            continue
        if not image_label_confidences.get(flickr_id):
            image_label_confidences[flickr_id] = {}
        label_name = row["LabelName"]
        confidence = row["Confidence"]
        image_label_confidences[flickr_id][label_name] = float(confidence)
    y_pred = []
    for flickr_id in subset_flickr_ids:
        image_preds = np.zeros(len(classes_encoder.tokens), dtype=np.float32)
        if image_label_confidences.get(flickr_id):
            # some images have no labels, seems to be when all labels are for classes that were marked as not being
            # present, likely implying that the confidences for all classes was less than 0, so we just leave them
            # with the array of 0 predictions
            for label, confidence in image_label_confidences[flickr_id].items():
                label_id = classes_encoder.encode(label)[0] - 1
                if label_id == 500:
                    # means that this is a label for a class that we aren't evaluating
                    continue
                image_preds[label_id] = confidence
        y_pred.append(image_preds)
    y_pred = np.array(y_pred)
    y_true = build_y_true(human_verified_subset_path, classes_encoder)
    categories = build_categories(label_names_file, classes_encoder)
    metrics = oid_challenge_evaluator_image_level(y_pred, y_true, categories)
    return metrics
