from oiv.common import get_train_val_test_ids
import flickrapi
import json
import time
import os
from common import load_csv_as_dict, write_rows_to_csv
from tqdm import tqdm
import urllib


def pre_process_tag(tag):
    new_user_tag = "+".join(urllib.parse.quote(sub_tag.lower()) for sub_tag in tag.split(" "))
    return new_user_tag


class Extend:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.flickr = flickrapi.FlickrAPI(api_key, api_secret, cache=True)
        self.last_call = 0

    def _get_user_tags(self, flickr_id):
        time_since_last = time.time() - self.last_call
        if time_since_last < 1:
            time.sleep(time_since_last)
        self.last_call = time.time()
        try:
            raw_tags = self.flickr.tags.getListPhoto(api_key=self.api_key, photo_id=flickr_id, format="json")
            tags_dict = json.loads(raw_tags.decode('utf-8'))
            if tags_dict.get("photo"):
                tags = [tag_dict["raw"] for tag_dict in tags_dict["photo"]["tags"]["tag"]
                        if tag_dict["machine_tag"] == 0]
            else:
                tags = None
        except Exception:
            tags = None
        return tags

    def get_user_tags_for_validation_and_test(self, oiv_folder_yfcc, oiv_folder_raw, output_folder, retries=3):
        print("Gettings ids at intersection of oiv and yfcc")
        oiv_yfcc_ids = get_train_val_test_ids(oiv_folder_yfcc, flickr_ids=True)
        print("Getting ids outside intersection")
        oiv_ids = get_train_val_test_ids(oiv_folder_raw, flickr_ids=True)
        for subset in ["validation", "test"]:
            print("Getting user tags for {}".format(subset))
            subset_path = os.path.join(output_folder, "{}_user_tags.csv".format(subset))
            founds_ids = set()
            if os.path.exists(subset_path):
                found = load_csv_as_dict(subset_path, fieldnames=["flickr_id", "user_tags"], delimiter="\t")
                founds_ids = set([row["flickr_id"] for row in found])
                print("{} photos user tags already fetched".format(len(founds_ids)))
            missing = (oiv_ids[subset] - oiv_yfcc_ids[subset]) - founds_ids
            print("Getting user tags for {} photos".format(len(missing)))
            rows = []
            for flickr_id in tqdm(missing):
                tags = None
                curr_retries = retries
                while tags is None and curr_retries != 0:
                    tags = self._get_user_tags(flickr_id)
                    curr_retries -= 1
                if tags is None:
                    continue
                tags = ",".join([pre_process_tag(tag) for tag in tags])
                row = {"flickr_id": flickr_id, "user_tags": tags}
                rows.append(row)
                if len(rows) == 600:
                    write_rows_to_csv(rows, subset_path, fieldnames=["flickr_id", "user_tags"], mode="a",
                                      delimiter="\t")
                    rows = []
            if rows:
                write_rows_to_csv(rows, subset_path, fieldnames=["flickr_id", "user_tags"], mode="a",
                                  delimiter="\t")
