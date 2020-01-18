import re
import os
from common import load_csv_as_dict


def extract_image_id_from_flickr_static(static_url):
    """ Given a static url extract the image id

    Parameters
    ----------
    static_url : str
        Static url to photo, one of kind:
            https://farm{farm-id}.staticflickr.com/{server-id}/{id}_{secret}.jpg
                or
            https://farm{farm-id}.staticflickr.com/{server-id}/{id}_{secret}_[mstzb].jpg
                or
            https://farm{farm-id}.staticflickr.com/{server-id}/{id}_{o-secret}_o.(jpg|gif|png)

    Returns
    -------
    str
        Image id of url
    """

    pattern = r"(?:.*?\/\/?)+([^_]*)"
    image_id = re.findall(pattern, static_url)[0]
    return image_id


def get_train_val_test_flickr_ids(oiv_folder):
    """ Extract the flickr image ids for train, validation, and test in the OIV dataset

    Parameters
    ----------
    oiv_folder : str
        Path to folder containing open images csv's

    Returns
    -------
    dict of set
        Dict mapping "train", "validation", and "test" to sets, containing the flickr ids of images that belong to them
    """

    files = {"train": "train-images-boxable-with-rotation.csv", "validation": "validation-images-with-rotation.csv",
             "test": "test-images-with-rotation.csv"}
    image_ids = {}
    for subset, file_name in files.items():
        subset_csv = load_csv_as_dict(os.path.join(oiv_folder, file_name))
        subset_ids = set()
        for row in subset_csv:
            flickr_url = row["OriginalURL"]
            flickr_id = extract_image_id_from_flickr_static(flickr_url)
            subset_ids.add(flickr_id)
        image_ids[subset] = subset_ids
    return image_ids
