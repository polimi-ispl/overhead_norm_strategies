"""
Utilities for creating the training and validation splits needed for the fingerprint extractor
Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
Nicolò Bonettini - nicolo.bonettini@polimi.it
Sara Mandelli - sara.mandelli@polimi.it
Paolo Bestagini - paolo.bestagini@polimi.it
"""

# Libraries import #
from glob import glob
import numpy as np
import os
from sklearn.model_selection import train_test_split
from typing import List, Dict
import random


def make_train_val_split(acquisitions_dir: str, num_tiles_peracq: int=None, perc_train_val: float = 0.5,
                         seed: int = 42, png_tiles: bool=False) -> (List, List):
    """
    Create the lists dividing the patches from the different acquisitions in training and validation splits
    :param acquisitions_dir: str, path to the directory containing all the images divided by acquisition of provenance
    :param num_tiles_peracq: int, number of tiles to use from each acquisition for making the splits
    :param perc_train_val: float, percentage used to split the whole dataset in train and validation
    :param seed: int, random seed used for splitting
    :param png_tiles: bool, wheter to look for 8-bits PNG or 16-bits TIFF tiles
    :return: (List, List), with two lists of paths of the images to load
    """

    # Set the seed for shuffling
    np.random.seed(seed)

    # Check how many acquisitions are available
    all_acqs = sorted(os.listdir(acquisitions_dir))
    if not num_tiles_peracq:
        num_tiles_peracq = min([len(os.listdir(os.path.join(acquisitions_dir, dir))) for dir in all_acqs]) - 1

    # Split into train, val, test
    list_val, list_train = train_test_split(all_acqs, test_size=perc_train_val, random_state=seed)

    # File format to look for
    file_format = 'png' if png_tiles else 'tiff'

    # Going with validation first
    val_samples = []
    for product in list_val:

        # dev_id = dev_list[indexD]
        print('Acquisition %s' % product)

        img_product_list = sorted(glob(os.path.join(acquisitions_dir, product, '*.{}'.format(file_format))))
        if not len(img_product_list):
            raise RuntimeError('No tiles found here! Are you sure the directory is right?')
        # randomly extract N images:
        random.seed(seed)
        img_val_list = random.sample(img_product_list, num_tiles_peracq)
        val_samples.append(img_val_list)

    # Going with training now
    train_samples = []
    for product in list_train:
        print('Acquisition %s' % product)

        img_product_list = sorted(glob(os.path.join(acquisitions_dir, product, '*.{}'.format(file_format))))
        if not len(img_product_list):
            raise RuntimeError('No tiles found here! Are you sure the directory is right?')
        # randomly extract N images:
        random.seed(seed)
        img_train_list = random.sample(img_product_list, num_tiles_peracq)
        train_samples.append(img_train_list)

    return train_samples, val_samples


def make_dir_split(acquisitions_dir: str, num_tiles_peracq: int=None, seed: int = 42) -> Dict[str, List]:
    """
    Create a list split dividing the samples found from the different directories in the indicated path
    :param acquisitions_dir: str, path to the directory containing all the images divided by acquisition of provenance
    :param num_tiles_peracq: int, number of tiles to use from each acquisition for making the splits
    :param seed: int, random seed used for splitting
    :return: List, with two lists of paths of the images to load
    """

    # Set the seed for shuffling
    np.random.seed(seed)

    # Check how many acquisitions are available
    all_dirs = sorted(os.listdir(acquisitions_dir))

    # Check how many tiles are available if not specified by the user
    smallest_dir = min([len(glob(os.path.join(acquisitions_dir, dir, '*.tiff'))) for dir in all_dirs])
    if (not num_tiles_peracq) or (num_tiles_peracq > smallest_dir):
        num_tiles_peracq = min([len(glob(os.path.join(acquisitions_dir, dir, '*.tiff'))) for dir in all_dirs]) - 1

    # Going with validation first
    samples = dict()
    for product in all_dirs:

        # product_id = product_list[indexD]
        print('Directory %s' % product)

        img_product_list = sorted(glob(os.path.join(acquisitions_dir, product, '*.tiff')))
        if not len(img_product_list):
            raise RuntimeError('No tiles found here! Are you sure the directory is right?')
        # randomly extract N images:
        random.seed(seed)
        img_list = random.sample(img_product_list, num_tiles_peracq)
        samples[product] = img_list

    return samples


