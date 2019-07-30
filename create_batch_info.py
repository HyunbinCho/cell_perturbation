import os
import glob
import cv2
import numpy as np
import pandas as pd

from collections import defaultdict
from utils.load_dataset import *

sys.path.append("/home/hyunbin/git_repositories/rxrx1-utils")
import rxrx.io as rio

import yaml
import numba


@numba.jit(nopython=True, parallel=True)
def calculate_subfunc(image_array):
    mean = np.mean(image_array)
    std = np.std(image_array)
    return mean, std


def create_batch_info(datapath, outpath):
    """
    calculates of mean and stddev per batch samples only using Negative Control

    Examples of batch_info_dict

    HEPG2-01:
        - mean: (0.5, 0.5, 0.5, 0.5, 0.5, 0.5)   ------>   6-channel
        - std: (0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

    HEPG2-02:
        - mean: (0.499, 0.323, 0.276, 0.301, 0.399, 0.501)
        - std:: (0.501, 0.333, 0.255, 0.532, 0.444, 0.333)

    ...(skip)

    U2OS-01:
        - mean: (0.375, 0.376, 0.377, 0.378, 0.379, 0.380)
        - std: (0.222, 0.223, 0.224, 0.225, 0.226, 0.227)

    ...(skip)


    """
    #data_path = "/data2/cell_perturbation/"

    traindata = load_data_cell_perturbation(base_path=os.path.join(datapath, "train"))
    testdata = load_data_cell_perturbation(base_path=os.path.join(datapath, "test"))
    metadata = load_metadata()

    merged_data = merge_all_data_to_metadata([traindata, testdata], metadata)
    print(merged_data)
    temp_batchname_list = [i.split("_")[0] for i in merged_data.loc[:, 'id_code'].values.tolist()]
    batch_nameset = set(temp_batchname_list)
    batch_nameset = list(batch_nameset)

    batch_info_dict = defaultdict(dict)

    for batch_name in batch_nameset:
        print(batch_name)
        batch_info_dict[batch_name] = defaultdict(dict)
        batch_info_dict[batch_name]['mean'] = list()
        batch_info_dict[batch_name]['std'] = list()

        temp_df = merged_data[merged_data['id_code'].str.match(batch_name)]
        temp_df = temp_df[temp_df['well_type'] == 'negative_control']

        #iterates channel 1 ~ 6
        for channel in range(1, 7):
            temp_df_per_channel = temp_df.loc[:, 'c{}'.format(channel)]

            img_arr = np.array([cv2.imread(i) for i in temp_df_per_channel.values.tolist()])

            #calculates mean, std each from all pixel values
            #TODO: make a subfunction with numba
            mean, std = calculate_subfunc(img_arr)
            batch_info_dict[batch_name]['mean'].append(mean)
            batch_info_dict[batch_name]['std'].append(std)

    with open(outpath, 'w', encoding="utf-8") as yaml_file:
        dump = yaml.dump(batch_info_dict, default_flow_style=False, allow_unicode=True, encoding=None)
        yaml_file.write(dump)


if __name__ == "__main__":
    data_path = "/data2/cell_perturbation"

    create_batch_info(data_path, "./batch_info.yaml")