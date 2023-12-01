"""
Training script for fingerprint extractor.
The basic concept draws from the Noiseprint (https://github.com/grip-unina/noiseprint) and is the following:
every RGB product possesses a peculiar fingerprint given by the processing chain it has undergone.
To extract this fingerprint we employ a classic DnCNN (https://github.com/cszn/DnCNN), dividing each product in
tiles due to the high dimensionality in pixels of the original images.
From these tiles, during training we further extract patches in random position and process them with the network: the
goal is to make the fingerprint extracted from patches coming from the same acquisitions as similar as possible.
To do so we employ the distance based logistic (DBL, https://arxiv.org/abs/1608.00161) function, using as distance the
mean squared error (MSE).
We call the resulting network a satellite fingerprint extractor.

Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
Sriram Baireddy - sbairedd@purdue.edu
Paolo Bestagini - paolo.bestagini@polimi.it
Stefano Tubaro - stefano.tubaro@polimi.it
Edward J. Delp - ace@purdue.edu
"""

# Libraries import #

import sys
import os
import numpy as np
import keras
from isplutils.data import SatTilesScaler, SatProductMaxScaler, SatProductRobustScaler, DBLDataGenerator
from isplutils.network import generate_separable_fingerprint_extractor
from isplutils.common import make_fe_train_tag
from isplutils.losses import DistanceBasedLogisticLoss
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import argparse


# Helper functions and classes #


def train(train_data_generator: keras.utils.Sequence, valid_data_generator: keras.utils.Sequence,
          init_model_path: str, model_dir: str, learning_rate: float = 0.001, weight_reg: float = 0.0,
          num_epochs: int = 30, verbose: bool = True, train_batch_size: int = 2, num_pos: int = 2,
          num_tiles: int = 4, input_channels: int = 3, output_channels: int = 3):
    """
    Trains the model
    :param train_data_generator: keras.Sequence, training batches generator
    :param valid_data_generator: keras.Sequence, validation batches generator
    :param init_model_path: str, path to the weights initialising the model
    :param model_dir: str, directory where to store model and trained weights
    :param learning_rate: int, learning rate
    :param weight_reg: float, weight for the l2 regularization of the network's weights
    :param num_epochs: int, number of epochs to train
    :param verbose: int, `0` to suppress verbose output, `1` for verbose output
    :param train_batch_size: int, number of acquisition in each batch
    :param num_pos: int, number of positions to extract patches from
    :param num_tiles: int, number of tiles considered from each acquisition in the batch
    :param input_channels: int, number of channels in the input raster data
    :param output_channels: int, number of channels in the output fingerprint,
    :param separable_fp: bool, either to use separable convolutions in the fingerprint extractor
    :param depthwise_fp: bool, either to use depthwise convolutions in the fingerprint extractor
    :return: None, execute the training routine
    """
    model = generate_separable_fingerprint_extractor(kernel_regularizer_weight=weight_reg,
                                                         model_path=None, image_channels=input_channels,
                                                         output_channels=output_channels)
    # --- Count the initial weights --- #
    trainable_count = int(
        np.sum([keras.backend.count_params(p) for p in set(model.trainable_weights)]))  # count the trainable weights

    # --- path output files --- #
    os.makedirs(model_dir, exist_ok=True)  # Make model path
    model_file = os.path.join(model_dir, 'model.json')
    weights_file = os.path.join(model_dir, 'model_weights.h5')
    log_file = os.path.join(model_dir, 'model_log.csv')

    with open(log_file, 'a' if os.path.isfile(log_file) else 'w') as fid:
        fid.write('np:%d\n' % trainable_count)

    # --- Save Model ---
    with open(model_file, 'w') as json_file:
        json_file.write(model.to_json())

    # --- Define loss-function, optimizer and callbacks --- #
    optimizer = keras.optimizers.adam(lr=learning_rate)

    train_loss = DistanceBasedLogisticLoss(train_batch_size, num_pos, num_tiles)
    model.compile(optimizer=optimizer, loss=train_loss, metrics=[train_loss])

    # Define utilities during training
    callbacks_list = [
        # Save the weights only when the accuracy on validation-set improves
        keras.callbacks.ModelCheckpoint(weights_file, monitor='val_dbl', verbose=verbose,
                                        save_best_only=True, mode='min'),
        # Save a CSV file with info
        keras.callbacks.CSVLogger(log_file, separator=';', append=True),
        # Stop the training if the accuracy on validation-set does not improve for 30 epochs
        keras.callbacks.EarlyStopping(monitor='val_dbl', mode='min', patience=30,
                                      verbose=verbose),
        # Reduce the learning-rate if the metric on validation-set does not improve for 5 epochs
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5)]

    # --- TRAIN THE MODEL --- #
    history = model.fit_generator(generator=train_data_generator,
                                  epochs=num_epochs,
                                  use_multiprocessing=False,
                                  validation_data=valid_data_generator,
                                  validation_steps=len(valid_data_generator),
                                  callbacks=callbacks_list,
                                  verbose=verbose,
                                  shuffle=False,
                                  steps_per_epoch=len(train_data_generator))

    # --- Save the training history --- #
    history_file_path = os.path.join(model_dir, 'history.npy')
    np.save(history_file_path, history.history)


# ----------------------------------------- Main -----------------------------------------------------------------------
def main(config: argparse.Namespace):

    # --- GPU configuration --- #
    if config.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu  # set the GPU device
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print('tf    version:', tf.__version__)
    print('keras version:', keras.__version__)

    configSess = tf.ConfigProto()
    # Allowing the GPU memory to grow avoiding preallocating all the GPU memory
    configSess.gpu_options.allow_growth = True
    set_session(tf.Session(config=configSess))

    # --- Training hyperparameters parsing --- #
    train_batch_size = config.batch_size
    model_path = config.model_save_path
    patch_size = config.patch_size
    num_tiles_peracq = config.num_tiles_peracq
    batch_num_tiles_peracq = config.batch_num_tiles_peracq
    batch_num_pos_pertile = config.batch_num_pos_pertile
    norm = config.input_norm
    pos_const = config.pos_const
    output_fp_channels = config.output_fp_channels

    # --- Instantiate the DataGenerators --- #
    train_data_generator = DBLDataGenerator(batch_size=train_batch_size, patch_size=patch_size,
                                            data_dir=config.train_dir, split_seed=config.split_seed,
                                            num_iteration=config.num_iteration,
                                            num_pos_pertile=batch_num_pos_pertile,
                                            num_tiles_peracq=num_tiles_peracq,
                                            batch_num_tiles_peracq=batch_num_tiles_peracq,
                                            scaler_type=config.scaler_type,
                                            mean_scaling_strategy=config.mean_robust_scaling,
                                            input_norm=norm)
    valid_data_generator = DBLDataGenerator(batch_size=train_batch_size, patch_size=patch_size,
                                            data_dir=config.val_dir, split_seed=config.split_seed,
                                            num_iteration=config.num_iteration,
                                            num_pos_pertile=batch_num_pos_pertile,
                                            num_tiles_peracq=num_tiles_peracq,
                                            batch_num_tiles_peracq=batch_num_tiles_peracq,
                                            scaler_type=config.scaler_type,
                                            mean_scaling_strategy=config.mean_robust_scaling,
                                            input_norm=norm)

    # --- TRAINING --- #
    print('Starting training')
    # Create the directory where to store the model's weights first
    # --- create different folders depending on the model parameters --- #
    model_dir = os.path.join(model_path,
                             make_fe_train_tag(epochs=config.epochs, num_iter=config.num_iteration,
                                               lr=config.learning_rate,
                                               batch_size=train_batch_size, num_tiles_peracq=config.num_tiles_peracq,
                                               split_seed=config.split_seed, weight_reg=config.weight_reg,
                                               batch_num_tiles=batch_num_tiles_peracq, num_pos=batch_num_pos_pertile,
                                               norm=norm, pos_const=pos_const, suffix=config.suffix,
                                               scaler=config.scaler_type, mean_scaling=config.mean_robust_scaling,
                                               p_aug=config.p_aug, output_fp_channels=output_fp_channels,
                                               ))
    print('Model directory is {}'.format(model_dir))
    train(train_data_generator=train_data_generator,
          valid_data_generator=valid_data_generator,
          init_model_path=config.init_model_path,
          model_dir=model_dir,
          learning_rate=config.learning_rate, weight_reg=config.weight_reg,
          num_epochs=config.epochs, num_pos=batch_num_pos_pertile, num_tiles=batch_num_tiles_peracq,
          verbose=True, train_batch_size=train_batch_size, input_channels=config.input_fp_channels,
          output_channels=config.output_fp_channels)

    # Release memory
    keras.backend.clear_session()


if __name__ == '__main__':

    # Introduce arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--patch_size', type=int, default=48)
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--batch_size', help='Number of satellite per batch for training', type=int, default=10)
    parser.add_argument('--num_iteration', help='Number of iterations per epoch', type=int, default=128)
    parser.add_argument('--learning_rate', help='learning rate', type=float, default=0.0001)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=500)
    parser.add_argument('--init_model_path', type=str, default='models/fe_init_weights.h5')
    parser.add_argument('--train_dir', type=str, default='data/dataset_fe/train',
                        help='Path to the directory containing the training images divided per satellite')
    parser.add_argument('--val_dir', type=str, default='data/dataset_fe/val',
                        help='Path to the directory containing the validation images divided per satellite')
    parser.add_argument('--split_seed', type=int, default=42, help='Random seed used for train/val splitting')
    parser.add_argument('--num_tiles_peracq', type=int, default=200,
                        help='How many tiles per product to consider creating the train/val splits')
    parser.add_argument('--model_save_path', type=str, default='models/fingerprint_extractors')
    parser.add_argument('--batch_num_tiles_peracq', type=int, default=10, action='store',
                        help='Number of tiles taken for satellite product in the batch')
    parser.add_argument('--batch_num_pos_pertile', type=int, default=6, action='store',
                        help='Number of positions from which take the patches from each tile in the batch')
    parser.add_argument('--weight_reg', type=float, default=0, help='Weight for the l2 regularization of the '
                                                                    'network\'s weights')
    parser.add_argument('--scaler_type', type=str, help='Choose the scaler for the data. Choices are: '
                                                        '99th percentile robust scaler;'
                                                        '95th percentile robust scaler;'
                                                        'Maximum scaling using each band statistics.',
                        default='99th_percentile', choices=['99th_percentile', '95th_percentile', 'sat_max',
                                                            'sat_tiles_scaler'])
    parser.add_argument('--mean_robust_scaling', action='store_true',
                        help='Strategy for the robust scaling. Either use the scaler trained on the mean band signal '
                             'or not. WATCH OUT, it does not influence the input norm scaling!!!')
    parser.add_argument('--input_norm', type=str, choices=['absolute_scaling',
                                                           'max_scaling', 'min_max_scaling', 'uniform_scaling'],
                        default='max_scaling',
                        help='Either normalize the input in a [0, 1] range (absolute, max_abs) or [-1, 1] (min_max) or '
                             'to apply a uniform equalization. WATCH OUT, it does not influence the robust scalers!!!')
    parser.add_argument('--input_fp_channels', type=int, default=3,
                        help='Number of input channels in the fingerprint extractor')
    parser.add_argument('--output_fp_channels', type=int, default=3,
                        help='Number of output channels in the fingerprint extractor')
    parser.add_argument('--suffix', type=str, help='Suffix to add to the model\'s tag', default=None)

    config = parser.parse_args()

    # Call main
    print('Starting training the fingerprint extractor...')
    try:
        main(config)
    except Exception as e:
        print('Something happened! Error is {}'.format(e))
    print('Training done! Bye!')

    # Exit
    sys.exit()