import csv
import re

import sys

maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)


def load_csv_as_dict(csv_path, fieldnames=None):
    """ Loads the csv DictReader

    Parameters
    ----------
    csv_path : str
        Path to csv
    fieldnames : list of str
        List of fieldnames, if None then fieldnames are take from the first row

    Returns
    -------
    csv.DictReader
        DictReader object of path
    """

    delimiter = "\t"
    f = open(csv_path, encoding='utf8')
    c = csv.DictReader(f, fieldnames=fieldnames, delimiter=delimiter)
    return c


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
