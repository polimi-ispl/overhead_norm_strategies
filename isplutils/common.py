"""
Common utils functions
Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
Nicolò Bonettini - nicolo.bonettini@polimi.it
Sara Mandelli - sara.mandelli@polimi.it
Paolo Bestagini - paolo.bestagini@polimi.it
"""

import numpy as np
from pprint import pprint


def uint82float32(img: np.array) -> np.array:
    """
    Normalize a uint8 image as float with values between 0-1
    :param img: np.array, uint8 image to normalize
    :return: np.array, float32 array normalized
    """
    img = img.astype(np.float32)
    img -= img.min()
    img /= img.max()
    return img


# Create tag for saving fingerprint extractor
def make_fe_train_tag(epochs: int,
                      num_iter: int,
                      lr: float,
                      batch_size: int,
                      num_tiles_peracq: int,
                      split_seed: int,
                      batch_num_tiles: int,
                      num_pos: int,
                      scaler: str,
                      mean_scaling: bool,
                      norm: str,
                      weight_reg: float,
                      pos_const: bool,
                      suffix: str,
                      p_aug: float,
                      output_fp_channels: int,
                      ):

    # Training parameters and tag
    if scaler != 'sat_tiles_scaler':
        tag_params = dict(epochs=epochs,
                          num_iter=num_iter,
                          lr=lr,
                          size=batch_size,
                          num_tiles_peracq=num_tiles_peracq,
                          split_seed=split_seed,
                          batch_num_tiles=batch_num_tiles,
                          num_pos=num_pos,
                          scaler=scaler,
                          mean_scaling=mean_scaling,
                          weight_reg=weight_reg,
                          pos_const=pos_const,
                          suffix=suffix,
                          p_aug=p_aug,
                          output_fp_channels=output_fp_channels
                          )
    else:
        tag_params = dict(epochs=epochs,
                          num_iter=num_iter,
                          lr=lr,
                          size=batch_size,
                          num_tiles_peracq=num_tiles_peracq,
                          split_seed=split_seed,
                          batch_num_tiles=batch_num_tiles,
                          num_pos=num_pos,
                          scaler=scaler,
                          input_norm=norm,
                          mean_scaling=mean_scaling,
                          weight_reg=weight_reg,
                          pos_const=pos_const,
                          suffix=suffix,
                          p_aug=p_aug,
                          output_fp_channels=output_fp_channels
                          )

    print('Parameters')
    pprint(tag_params)
    tag = 'train_'
    tag += '_'.join(['-'.join([key, str(tag_params[key])]) for key in tag_params])
    print('Tag: {:s}'.format(tag))
    return tag


# Create tag for saving segmentation models
def make_seg_train_tag(net_class: str,
                       backbone: str,
                       epochs: int,
                       batch_size: int,
                       perc_test_split: float,
                       perc_val_split: float,
                       lr: float,
                       seed: int,
                       suffix: str,
                       debug: bool,
                       ):

    # Training parameters and tag
    tag_params = dict(net=net_class,
                      backbone=backbone,
                      epochs=epochs,
                      lr=lr,
                      size=batch_size,
                      perc_test_split=perc_test_split,
                      perc_val_split=perc_val_split,
                      seed=seed
                      )
    print('Parameters')
    pprint(tag_params)
    tag = 'debug_' if debug else ''
    tag += '_'.join(['-'.join([key, str(tag_params[key])]) for key in tag_params])
    if suffix is not None:
        tag += '_' + suffix
    print('Tag: {:s}'.format(tag))
    return tag