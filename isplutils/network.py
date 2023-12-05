"""
Network definition file

All the networks deployed in the detector are defined in this file

Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
Sriram Baireddy - sbairedd@purdue.edu
Paolo Bestagini - paolo.bestagini@polimi.it
Stefano Tubaro - stefano.tubaro@polimi.it
Edward J. Delp - ace@purdue.edu
"""

# Libraries import #
import keras.regularizers
import numpy as np
import tensorflow as tf
from keras.models import *
from keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, BatchNormalization, Activation, Lambda, Subtract, Dropout
from pathlib import Path
import os
import json


# Network creator helpers #

def generate_separable_fingerprint_extractor(depth: int = 17, first_depthwise_multiplier: int = 22, filters: int = 48,
                                             image_channels: int = 1, output_channels: int = 1, use_bnorm: bool = True,
                                             kernel_regularizer_weight: float = 0.0, model_path: str = None) -> Model:
    """
    Helper function for the creation of an instance of the fingerprint extractor (FE) definition
    It is an implementation inspired by the DnCNN (https://github.com/cszn/DnCNN), but using a first depthwise followed by
    separable convolutions instead of standard convolutions, to better match the different information in the 3-bands
    of RGB overhead imagery.
    :param depth: int, number of layers
    :param first_depthwise_multiplier: int, number of filters applied per input channel in first layer
    :param image_channels: int, number of color channels
    :param output_channels: int, number of bands in the output fingerprint
    :param use_bnorm: bool, whether to use batch normalization (True) or not (False)
    :param kernel_regularizer_weight: float, 0-1 value for the l2 penalty norm on the layers' weights
    :param model_path: str, path to model weights in case we using a pretrained model
    :return: keras.models.Model instance of the FE
    """

    # --- Define the architecture --- #
    layer_count = 0
    input = Input(shape=(None, None, image_channels), name='input' + str(layer_count))
    # 1st layer, DepthwiseConv+relu
    layer_count += 1
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), depthwise_initializer='Orthogonal', padding='same',
                        depth_multiplier=first_depthwise_multiplier, name='depth_conv' + str(layer_count),
                        depthwise_regularizer=keras.regularizers.l2(kernel_regularizer_weight))(input)
    layer_count += 1
    x = Activation('relu', name='relu' + str(layer_count))(x)
    # depth-2 layers, DepthwiseConv+BN+relu
    for i in range(depth - 2):
        layer_count += 1
        x = SeparableConv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), depthwise_initializer='Orthogonal', depth_multiplier=1,
                            padding='same', use_bias=False, name='depth_conv' + str(layer_count),
                            depthwise_regularizer=keras.regularizers.l2(kernel_regularizer_weight))(x)
        if use_bnorm:
            layer_count += 1
            x = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.0001, name='bn' + str(layer_count))(x)
        layer_count += 1
        x = Activation('relu', name='relu' + str(layer_count))(x)
    # last layer, Conv (we process all the separate filters to get back to a 1-channel or 3-channel output)
    layer_count += 1
    x = SeparableConv2D(filters=output_channels, kernel_size=(3, 3), strides=(1, 1), depthwise_initializer='Orthogonal',
                        pointwise_initializer='Orthogonal',
                        padding='same', use_bias=False, name='conv' + str(layer_count),
                        depthwise_regularizer=keras.regularizers.l2(kernel_regularizer_weight),
                        pointwise_regularizer=keras.regularizers.l2(kernel_regularizer_weight))(x)
    layer_count += 1
    # x = Subtract(name='subtract' + str(layer_count))([inpt, x])  # input - noise
    model = Model(inputs=input, outputs=x)

    # --- Load the pretrained weights --- #

    if (model_path is not None) and (image_channels == 1):
        # 1-channel denoiser model
        model.load_weights(model_path, by_name=True)
        print('loaded ', model_path)
    elif (model_path is not None) and (image_channels != 1):
        # multi-channel denoiser model
        # what changes with respect to the 1-channel model is the first and last layers
        # we are going to load the 1-channel model, and then set all the weights matching the same shape
        # from the pretrained model
        model_1channel = generate_separable_fingerprint_extractor(image_channels=1, model_path=model_path)
        weights_list = model_1channel.get_weights().copy()  # get the weights list
        for i, weights in enumerate(weights_list):
            # set the weights not matching the shape as the random initialized ones
            if model.get_weights()[i].shape != weights.shape:
                weights_list[i] = model.get_weights()[i].copy()
        model.set_weights(weights_list)

    return model
