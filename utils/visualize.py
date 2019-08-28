import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import matplotlib.pyplot as plt
#import seaborn as sns

import cv2
import math

sys.path.append("/home/hyunbin/utils/rxrx1-utils")
import rxrx.io as rio


"""

to install library "rxrx",
refer to "https://github.com/recursionpharma/rxrx1-utils"

"""


def plot_channels(data):
    """
    visualize a set of images (6 channels) each in subplot (2 X 3)

    only use this function within jupyter notebook!!

    input : a list of images that is made by the same position, differently focused
            the length of th list should be 6

    input can be easily created as using the function 'load_dataset_cell_perturbation'

    ex)
    plot_channels(dataset[10])

    -AchB 2019.07.22-
    """

    fig = plt.figure(figsize=(20, 10))

    for i in range(len(data)):
        tmp_img = plt.imread(data[i])
        plt.subplot(2, 3, i + 1)
        plt.imshow(tmp_img, cmap='gray', figure=fig)
        plt.title("channel {}".format(i + 1), figure=fig)

    #return None


def plot_rgb(data, figsize=10):
    """
    convert 6-channels to RGB (3-channel)
    and visualize the figure.

    """
    dirs = data[0].split("/")
    base_plates = [dirs.index(i) for i in dirs if i.startswith("Plate")][0]
    pre = "_".join(dirs[base_plates-1: base_plates+1])
    suf = "_".join(os.path.basename(data[0]).split("_")[:2])

    imgname = "_".join((pre, suf))

    plt.figure(figsize=(figsize, figsize))

    img_6channel = rio.load_images_as_tensor(data)
    img_as_rgb = rio.convert_tensor_to_rgb(img_6channel)
    plt.title(imgname)
    plt.imshow(img_as_rgb)


#preprocessed image (custom)
def plot_channels_with_clahe(data, cliplimit, tilesize):
    """
    visualize a set of images (6 channels) each in subplot (2 X 3)
    preprocessing images with CLAHE (a kind of histogram equalization)

    ex)
    compare between
    cliplimit = 3, tilesize= 3
    cliplimit = 15, tilesize = 3
    cliplimit = 15, tilesize = 15
    cliplimit = 15, tilesize = 75


    """
    temp_img_list = [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in data]
    clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(tilesize, tilesize))
    clahe_img_list = [clahe.apply(i) for i in temp_img_list]

    plt.figure(figsize=(20, 10))

    for i in range(len(temp_img_list)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(clahe_img_list[i], cmap='gray')
        plt.title("channel {}".format(i + 1))


def plot_cell_sirna_rgb(filtered_data, figsize=20):
    """
    plot all data in a row (Actually mutliple rows)

    :param filtered_data: dataframe, after processed with 'filter_data_with_cell_sirna'
    :param figsize: column size
    :return:
    """
    data_num = len(filtered_data)
    row_num = math.ceil((data_num // 4))

    plt.figure(figsize=(figsize, figsize*row_num * 0.25))

    for i in range(data_num):
        plt.subplot(row_num, 4, i+1)
        img_paths = filtered_data.iloc[i, -6:].tolist()

        img_6channel = rio.load_images_as_tensor(img_paths)
        img_as_rgb = rio.convert_tensor_to_rgb(img_6channel)
        plt.title(filtered_data.iat[i, 0])
        plt.imshow(img_as_rgb)


def plot_cell_sirna_channels(filtered_data, figsize=20):
    """
    plot all data in a row (Actually mutliple rows)

    :param filtered_data: dataframe, after processed with 'filter_data_with_cell_sirna'
    :param figsize: column size
    :return:
    """
    data_num = len(filtered_data) * 6
    row_num = math.ceil((data_num // 6))

    plt.figure(figsize=(figsize, figsize*row_num * 0.25))

    for i in range(len(filtered_data)):
        img_paths = filtered_data.iloc[i, -6:].tolist()

        for j in range(len(img_paths)):
            tmp_img = plt.imread(img_paths[j])
            plt.subplot(row_num, 6, i*6 + j + 1)
            plt.title("{}_c{}".format(filtered_data.iloc[i, 0],(j+1)))

            plt.imshow(tmp_img, cmap='gray')






