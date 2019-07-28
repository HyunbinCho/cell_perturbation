import os
import glob

import pandas as pd
import numpy as np


def filter_data_with_cell_sirna(merged_data, cell, sirna):
    """
    return filtered DataFrame from merged_data (produced from 'load_dataset.merge_all_data_to_metadata')

    --------------
    arguments

    cell: str or list
    sirna: int or str or list

    ---------------
    Examples

    filtered_HEPG2_998 = filter_data_with_cell_sirna(merged_data, 'HEPG2', 998)

    filtered_some_condition = filter_data_with_cell_sirna(merged_data, ['HEPG2', 'HUVEC'], [i for i in range(1108, 1138)])  #1108 ~ 1137 : positive control
    """
    if isinstance(cell, list):
        cell = '|'.join(cell)

    if isinstance(sirna, list):
        sirna = '|'.join(map(str, sirna))
    else:
        sirna = str(sirna)

    filtered = merged_data[merged_data['id_code'].str.contains(cell)]
    filtered = filtered[filtered['sirna'].str.contains(sirna)]

    return filtered