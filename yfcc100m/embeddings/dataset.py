from yfcc100m.load import load_csv_as_dict
from tqdm import tqdm

FIELDS = ["LineNumber", "ID", "Hash", "UserNSID", "UserNickname", "DateTaken", "DateUploaded",
          "CaptureDevice", "Title", "Description", "UserTags", "MachineTags", "Longitude", "Latitude",
          "LongLatAcc", "PageURL", "DownloadURL", "LicenseName", "LicenseURL", "ServerIdentifier",
          "FarmIdentifier", "Secret", "OriginalSecret", "OriginalExtension", "Video"]


def count_user_tags(path):
    dataset = load_csv_as_dict(path, fieldnames=FIELDS)
    tag_counts = {}
    for row in tqdm(dataset):
        tags = row["UserTags"].split(",")
        for tag in tags:
            if not tag_counts.get(tag):
                tag_counts[tag] = 0
            tag_counts[tag] += 1
    return tag_counts
