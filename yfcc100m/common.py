import re
import os
from common import load_csv_as_dict, write_rows_to_csv


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


def get_dataset_fields():
    """ Return the fields for dataset file

    Returns
    -------
    list of str
        Fields for dataset file
    """

    return ["LineNumber", "ID", "Hash", "UserNSID", "UserNickname", "DateTaken", "DateUploaded",
            "CaptureDevice", "Title", "Description", "UserTags", "MachineTags", "Longitude", "Latitude",
            "LongLatAcc", "PageURL", "DownloadURL", "LicenseName", "LicenseURL", "ServerIdentifier",
            "FarmIdentifier", "Secret", "OriginalSecret", "OriginalExtension", "Video"]


def get_autotag_fields():
    """ Return the fields for autotag file

    Returns
    -------
    list of str
        Fields for autotag file
    """

    return ["ID", "PredictedConcepts"]


def get_places_fields():
    """ Return the fields for places file

    Returns
    -------
    list of str
        Fields for places file
    """

    return ["ID", "Places"]


def extend_dataset(yfcc_dir, validation_extend_path, test_extend_path):
    """ Extends the YFCC dataset using the validation and test expansions produced by oiv.extend. Appends to
        yfcc100m_dataset, but if yfcc100m_places and/or yfcc100m_autotags is present in the folder too, also adds
        empty rows to them so that the lines across all files correspond to the same images

    Parameters
    ----------
    yfcc_dir : str
        Directory to YFCC files to extends
    validation_extend_path : str
        Directory to validation user tags extension file produced by oiv.extend
    test_extend_path : str
        Directory to test user tags extension file produced by oiv.extend

    """

    dataset_rows = []
    autotags_rows = []
    places_rows = []
    for path in [validation_extend_path, test_extend_path]:
        extend_csv = load_csv_as_dict(path, delimiter="\t", fieldnames=["flickr_id", "user_tags"])
        for row in extend_csv:
            dataset_row = {field: "" for field in get_dataset_fields()}
            autotags_row = {field: "" for field in get_autotag_fields()}
            places_row = {field: "" for field in get_places_fields()}
            for yfcc_row in [dataset_row, autotags_row, places_row]:
                yfcc_row["ID"] = row["flickr_id"]
            dataset_row["UserTags"] = row["user_tags"]
            dataset_row["Video"] = "0"
            dataset_rows.append(dataset_row)
            autotags_rows.append(autotags_row)
            places_rows.append(places_row)
    dataset_path = os.path.join(yfcc_dir, "yfcc100m_dataset")
    if os.path.exists(dataset_path):
        write_rows_to_csv(dataset_rows, dataset_path, fieldnames=get_dataset_fields(), mode="a", delimiter="\t")
    autotags_path = os.path.join(yfcc_dir, "yfcc100m_autotags")
    if os.path.exists(autotags_path):
        write_rows_to_csv(autotags_rows, autotags_path, fieldnames=get_autotag_fields(), mode="a", delimiter="\t")
    places_path = os.path.join(yfcc_dir, "yfcc100m_places")
    if os.path.exists(places_path):
        write_rows_to_csv(places_rows, places_path, fieldnames=get_places_fields(), mode="a", delimiter="\t")
