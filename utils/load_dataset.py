import sys
import os
import glob
from collections import defaultdict

import random

import pandas as pd
import numpy as np

sys.path.append("/home/hyunbin/git_repositories/rxrx1-utils")
import rxrx.io as rio

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.resnet import resnet18, resnet50
#from torchvision.models import mobilenet torch의 v1.2.0 으로 업데이트 필요

import torchvision.transforms.functional as TF
from torchvision import transforms


# def load_data_cell_perturbation(base_path="/data2/cell_perturbation/train/"):
#     result = []
#
#     wells = os.listdir(base_path)
#     for well in wells:
#         plates = os.listdir(os.path.join(base_path, well))
#         for plate in plates:
#             images = glob.glob(os.path.join(base_path, well, plate, "*"))
#             temp_dict = defaultdict(list)
#
#             for image in images:
#                 basename = os.path.basename(image)
#                 elem = basename.split("_")
#                 setname = "_".join(elem[:2])
#                 # channel = elem[2]
#                 temp_dict[setname].append(image)
#
#             for val in temp_dict.values():
#                 # if len(val) != 6: continue
#                 result.append(sorted(val))
#
#     return result


def load_data_cell_perturbation(base_path="/data2/cell_perturbation/train/"):
    """
        return a dictionary
        -keys : sample info
        -values : a list of sample file paths

        -----------------------------------
        (sample example)


        """
    dataset_dict = dict()

    wells = os.listdir(base_path)
    for well in wells:
        plates = os.listdir(os.path.join(base_path, well))
        for plate in plates:
            images = glob.glob(os.path.join(base_path, well, plate, "*"))
            temp_dict = defaultdict(list)

            for image in images:
                #dirs = os.path.split(image)
                normpath = os.path.normpath(image)
                dirs = normpath.split(os.sep)
                basename = dirs[-1]
                elem = basename.split("_")
                pre = dirs[-3]
                plate = dirs[-2][-1]
                suf = "_".join(elem[:2])
                # channel = elem[2]
                sample_info = "{}_{}_{}".format(pre, plate, suf)
                temp_dict[sample_info].append(image)

            for key, val in temp_dict.items():
                # if len(val) != 6: continue
                dataset_dict[key] = sorted(val)

    return dataset_dict


def load_metadata():
    """
    returns metadata as pandas.DataFrame
    """
    #metadata = rio.combine_metadata()
    metadata = pd.read_pickle("/data2/cell_perturbation/metadata.pickle")

    return metadata


def merge_all_data_to_metadata(datalist, metadata):
    """
    returns merged data

    ------------------------------
    datalist : list, each element produced from 'load_data_cell_perturbation' function
    metadata : pandas.DataFrame, produced from 'load_metadata' function

    -------------------------------
    (usage ex)
    train = load_data_cell_perturbation(base_path="/data2/cell_perturbation/train/")
    test = load_data_cell_perturbation(base_path="/data2/cell_perturbation/test/")
    metadata = load_metadata()

    merged_data = merge_all_data_to_metadata([train, test], metadata)
    --------------------------------
    """
    concat_data_df = None
    for i, dataset in enumerate(datalist):
        dataset_df = pd.DataFrame(dataset).transpose()
        dataset_df.columns = ['c{}'.format(i) for i in range(1, 7)]

        rev_id_code = []
        rev_site = []
        for i in dataset_df.index.values.tolist():
            elem = i.split("_")

            temp_id = "_".join(elem[:-1])
            temp_site = int(elem[-1][1])

            rev_id_code.append(temp_id)
            rev_site.append(temp_site)

        dataset_df['id_code'] = rev_id_code
        dataset_df['site'] = rev_site

        dataset_df.reset_index(inplace=True)
        dataset_df.drop('index', axis=1, inplace=True)

        if i == 0:
            concat_data_df = dataset_df
        else:
            concat_data_df = pd.concat([concat_data_df, dataset_df])

    merged_data = metadata.merge(concat_data_df, how='left', on=['id_code', 'site'])
    merged_data['sirna'] = [str(i) for i in merged_data['sirna'].values.tolist()]

    return merged_data


def load_net(net_name, pretrained_path=None, zoo_pretrained=False):
    if net_name == 'resnet18':
        net = resnet18(zoo_pretrained)

    elif net_name == 'resnet50':
        net = resnet50(zoo_pretrained)

    elif net_name == 'mobilenet':
        raise NotImplementedError

    elif net_name == 'vggnet':
        raise NotImplementedError

    else:
        raise ValueError("invalid net_name : {}".format(net_name))

    if eval(pretrained_path) is not None:
        net.load_state_dict(torch.load(pretrained_path))
        print("pretrained {} weights will be used".format(pretrained_path))

    return net


class TrainDatasetRecursion(Dataset):
    def __init__(self, merged_data, args, isNormalize, isTrain, train_ratio=0.8, seed=10):
        df = merged_data[merged_data['dataset'] == 'train']

        if isNormalize:
            df = self.standardize_with_nc(df)

        #extract only treatments
        df = df[df['well_type'] == 'treatment']

        #split train & validation with fixed seed
        np.random.seed(seed)
        df = df.iloc[np.random.permutation(len(df))]

        train_num = int(train_ratio * len(df))

        if isTrain:
            train_df = df.iloc[:train_num, :]
            self.len_dataset = len(train_df)
            self.input_array = np.array(train_df.iloc[:, -6:])
            self.output_array = np.array(train_df.loc[:, 'sirna'])
        else:
            val_df = df.iloc[train_num:, :]
            self.len_dataset = len(val_df)
            self.input_array = np.array(val_df.iloc[:, -6:])
            self.output_array = np.array(val_df.loc[:, 'sirna'])

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):

        return (self.input_array[index], self.output_array[index])

    def transform(self, img, batch_num):
        # Random horizontal flipping
        # if random.random() > 0.5:
        #     img = TF.hflip(img)
        #     mask = TF.hflip(mask)
        #
        # # Random vertical flipping
        # if random.random() > 0.5:
        #     img = TF.vflip(img)
        #     mask = TF.vflip(mask)
        #
        # # Random rotation
        # if random.random() > 0.25:
        #     angle = random.choice([90, 180, 270])
        #     img = TF.rotate(img, angle)
        #     mask = TF.rotate(mask, angle)

        # Transform to tensor
        img = TF.to_tensor(img)


        normalization = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        img = normalization(img)

        return img

    def standardize_with_nc(self, df):
        """
        alleviates the batch effect & (plate effect if possible)

        first, put together the same cell samples of 'negative control'.
        next, standardize with sklearn.normalize or something(torch?) only for the subset above.
        (creates a Scaler with negative control and fit treatment samples to it)


        iterates the same process to all cases

        """
        treatment_df = df[df['well_type'] == 'treatment']
        nc_df = df[df['well_type'] =='negative_control']

        standardized_df = None
        return standardized_df


class TestDatasetRecursion(Dataset):
    def __init__(self, merged_data, args, isNormalize):
        df = merged_data[merged_data['dataset'] == 'test']

        if isNormalize:
            df = self.standardize_with_nc(df)

        # extract only treatments
        test_df = df[df['well_type'] == 'treatment']

        self.len_dataset = len(test_df)
        self.input_array = np.array(test_df.iloc[:, -6:])

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        return self.input_array[index]

    def standardize_with_nc(self, df):
        """
        alleviates the batch effect & (plate effect if possible)

        first, put together the same cell samples of 'negative control'.
        next, standardize with sklearn.normalize or something(torch?) only for the subset above.
        (creates a Scaler with negative control and fit treatment samples to it)


        iterates the same process to all cases

        """

        standardized_df = None
        return standardized_df