import re


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
