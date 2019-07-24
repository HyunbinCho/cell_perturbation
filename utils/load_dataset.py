import sys
import os
import glob
from collections import defaultdict

import pandas as pd
import numpy as np

# sys.path.append("/home/hyunbin/git_repositories/rxrx1-utils")
# import rxrx.io as rio


def load_data_cell_perturbation(base_path="/data2/cell_perturbation/train/"):
    result = []

    wells = os.listdir(base_path)
    for well in wells:
        plates = os.listdir(os.path.join(base_path, well))
        for plate in plates:
            images = glob.glob(os.path.join(base_path, well, plate, "*"))
            temp_dict = defaultdict(list)

            for image in images:
                basename = os.path.basename(image)
                elem = basename.split("_")
                setname = "_".join(elem[:2])
                # channel = elem[2]
                temp_dict[setname].append(image)

            for val in temp_dict.values():
                # if len(val) != 6: continue
                result.append(sorted(val))

    return result


def load_metadata(root_path="/data2/cell_perturbation"):
    metadata_train = glob.glob(os.path.join(root_path, "train*.csv"))

    meta_train_df = None
    meta_test_df = None

    for i, datapath in metadata_train:
        if i == 0:
            result_df = pd.read_csv(datapath)
        else:
            temp_df = pd.read_csv(datapath)
            pd.concat([result_df])