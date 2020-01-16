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
