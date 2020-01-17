from yfcc100m.load import load_csv_as_dict
from tqdm import tqdm

FIELDS = ["ID", "PredictedConcepts"]


def unique_tags(path):
    """ Gets the unique tags from the auto tags file

    Parameters
    ----------
    path : str
        Path to autotags file

    Returns
    -------
    set of str
        Unique tags
    """

    autotags = load_csv_as_dict(path, fieldnames=FIELDS)
    concepts = set()
    for row in tqdm(autotags):
        predicted_concepts = [predicted_concept.split(":")[0] for predicted_concept in
                              row["PredictedConcepts"].split(",")]
        for concept in predicted_concepts:
            concepts.add(concept)
    return concepts


def kept_classes(class_path):
    """ Get the list of classes to keep

    Parameters
    ----------
    class_path : str
        Path to class file of format "tag,{1 (keep) or 0 (discard)}"

    Returns
    -------
    list str
        List of tags to keep
    """

    f = open(class_path, "r")
    lines = f.readlines()
    classes = []
    for line in lines:
        line = line.rstrip()
        name, kept = line.split(",")
        if kept == '1':
            classes.append(name)
    return classes


def class_counts(path):
    """ Count the number of examples for each class

    Parameters
    ----------
    path : str
        Path to autotags file

    Returns
    -------
    dict of str -> int
        Counts for each class
    """

    autotags = load_csv_as_dict(path, fieldnames=FIELDS)
    counts = {}
    for row in tqdm(autotags):
        predicted_concepts = [predicted_concept.split(":")[0] for predicted_concept in
                              row["PredictedConcepts"].split(",")]
        for concept in predicted_concepts:
            if not counts.get(concept):
                counts[concept] = 0
            counts[concept] += 1
    return counts
