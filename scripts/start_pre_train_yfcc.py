"""Launches 3 pre-train jobs on YFCC100M with 3 different seeds"""

import pathlib
import os
import re

proj_root = pathlib.Path(__file__).parent.parent.absolute()
pre_train_path = os.path.join(proj_root, "milano_configs/embeddings_pretrain_run.sh")
original_script = "".join(open(pre_train_path, "r").readlines())

for i in range(0, 3):
    script = re.sub('export RANDOM_SEED="0"', original_script, 'export RANDOM_SEED="{}"'.format(i))
    output_script = "pre_train_{}.sh".format(i)
    open(output_script, "w").write(script)
    os.system("sbatch {}".format(output_script))
