from oiv.extend import Extend
from yfcc100m.common import extend_dataset
import os

api_key = "YOUR API KEY"
api_secret = "YOUR API SECRET"
oiv_folder_yfcc = "C:\\Users\\kiera\\OneDrive - University of Edinburgh\\Every Day Files\\Documents\\" \
                  "Microsoft Office\\Word\\Homework\\University\\Year 4\\Honours Project\\Open Images\\v5\\" \
                  "Image Level\\Extended"
oiv_folder_raw = "C:\\Users\\kiera\\OneDrive - University of Edinburgh\\Every Day Files\\Documents\\" \
                 "Microsoft Office\\Word\\Homework\\University\\Year 4\\Honours Project\\Open Images\\v5\\" \
                 "Image Level"
output_folder = "C:\\Honors Project\\ExtendedUserTags"
yfcc_dir = "C:\\Honors Project\\YFCC100M"

ex = Extend(api_key, api_secret)
# this takes approx 12 hours
ex.get_user_tags_for_validation_and_test(oiv_folder_yfcc=oiv_folder_yfcc, oiv_folder_raw=oiv_folder_raw,
                                         output_folder=output_folder)
extend_dataset(yfcc_dir=yfcc_dir, validation_extend_path=os.path.join(output_folder, "validation_user_tags.tsv"),
               test_extend_path=os.path.join(output_folder, "test_user_tags.tsv"))
