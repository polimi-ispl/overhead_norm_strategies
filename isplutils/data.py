"""
Various data utilities

Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
Nicolò Bonettini - nicolo.bonettini@polimi.it
Sara Mandelli - sara.mandelli@polimi.it
Paolo Bestagini - paolo.bestagini@polimi.it
Stefano Tubaro - stefano.tubaro@polimi.it
"""

# Libraries import #
import numpy as np
from keras.utils import np_utils, Sequence
import cv2
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from isplutils.split import make_dir_split
import rasterio
import ntpath
from PIL import Image
from joblib import load
import os
from abc import ABC, abstractmethod
import imgaug.augmenters as iaa
import skimage.exposure
from typing import List


# Helpers classes #
class Scaler(ABC):
    """
    Abstract class for the different scalers we are going to use
    """

    @abstractmethod
    def normalize_band(self, img: np.array, band:str) -> np.array:
        raise NotImplemented

    @abstractmethod
    def normalize_product(self, img: np.array) -> np.array:
        raise NotImplemented


class SatProductMaxScaler(Scaler):
    """
        Simple class to hold all the statistics for a specific satellite product
    """

    def __init__(self, *, product: str = None):
        self.product = product
        self.min = dict()
        self.max = dict()

    def normalize_band(self, img: np.array, band: str) -> np.array:
        """
        Normalize a single band of a satellite product

        :param img: np.array, the band to normalize
        :param band: str, the band to normalize
        """
        return img / self.max[band]

    def normalize_product(self, img: np.array, mean_scaling: bool = True) -> np.array:
        """
        Normalize all bands of a satellite product

        :param img: np.array, the satellite product to normalize
        :param mean_scaling: bool, whether to use the statistics from the mean intensity band
        """
        # Find the channel dimension as the lower value in raster shape
        min_dim = np.argmin(img.shape)
        if img.shape[min_dim] > 3:
            raise RuntimeError('Normalization works only for RGB images, sorry!')
        else:
            img = img.astype(np.float32)
            # Move the channel as the first dimension
            img = np.moveaxis(img, min_dim, 0)
            for i, band in enumerate(['Red', 'Green', 'Blue']):
                if mean_scaling:
                    img[i] /= self.max['Mean']
                else:
                    img[i] /= self.max[band]
                # Report everything between 0 and 1
                # img[i] = (img[i]-img[i].min())/(img[i].max()-img[i].min())
        # Move the channel as the last dimension
        img = np.moveaxis(img, 0, -1)
        return img

    def open_and_normalize_product(self, path: str, method: bool = True) -> np.array:
        """
        As the name suggests
        """
        # Open the product
        with rasterio.open(path, 'r') as src:
            img = src.read()
        # Normalize it
        img = self.normalize_product(img, method)
        return img


class SatProductRobustScaler(Scaler):
    """
    Simple class to hold all the RobustScalers for a specific satellite product
    """

    def __init__(self, *, product: str = None, quantile_range: List[float] = (0, 99)):
        self.product = product
        self.scalers = dict()
        self.min = dict()
        self.max = dict()
        self.quantile_range = quantile_range

    def normalize_band(self, img: np.array, band: str) -> np.array:
        """
        Normalize a single band of a satellite product

        :param img: np.array, the band to normalize
        :param band: str, the band to normalize
        """
        img = self.scalers[band].transform(img.reshape(-1, 1)).reshape(img.shape)
        return ((img - self.min[band]) / (self.max[band] - self.min[band]))

    def normalize_product(self, img: np.array, mean_scaling: bool = True) -> np.array:
        """
        Normalize all bands of a satellite product

        :param img: np.array, the product to normalize
        :param mean_scaling: bool, whether to use mean scaling for normalizing or not
        """
        # Find the channel dimension as the lower value in raster shape
        min_dim = np.argmin(img.shape)
        if img.shape[min_dim] > 3:
            raise RuntimeError('Normalization works only for RGB images, sorry!')
        else:
            img = img.astype(np.float32)
            # Move the channel as the first dimension
            img = np.moveaxis(img, min_dim, 0)
            if mean_scaling:
                for i in range(img.shape[0]):
                    img[i] = self.scalers['Mean'].transform(img[i].reshape(-1, 1)).reshape(img[i].shape)
                    img[i] = ((img[i] - self.min['Mean']) / (self.max['Mean'] - self.min['Mean']))
                    # Report everything between 0 and 1
                    # img[i] = (img[i] - img[i].min()) / (img[i].max() - img[i].min())
            else:
                for i, band in enumerate(['Red', 'Green', 'Blue']):
                    img[i] = self.scalers[band].transform(img[0].reshape(-1, 1)).reshape(img[0].shape)
                    img[i] = ((img[i] - self.min[band]) / (self.max[band] - self.min[band]))
                    # Report everything between 0 and 1
                    # img[i] = (img[i] - img[i].min()) / (img[i].max() - img[i].min())
            # Move the channel as the last dimension
            img = np.moveaxis(img, 0, -1)
        return img

    def open_and_normalize_product(self, path: str, method: bool = True) -> np.array:
        # Open the product
        with rasterio.open(path, 'r') as src:
            img = src.read()
        # Normalize it
        img = self.normalize_product(img, method)
        return img

    def normalize_unknown_product(self, img: np.array, mean_scaling: bool) -> np.array:
        """
        In case we don't know the product from which the tile comes from, we can proceed in a two step fashion:
            1. Look at the image dtype
            2. Look at the image size
        If the image size is < 4 megapixels, we can't trust its statistics for normalizing it.
        If instead it is > 4 megapixels, we can compute a RobustScaler on its statistics
        """
        # Find channel dimension as minimum value in raster shape
        min_dim = np.argmin(img.shape)
        if img.shape[min_dim] > 3:
            raise RuntimeError('Normalization works only for RGB images, sorry!')
        else:
            # Move channel dimension as first dimension
            img = np.moveaxis(img, min_dim, 0)
            if img.dtype == np.uint8:
                img = img.astype(np.float32)
                img /= 255
            else:
                # if img[1:-1].ravel().shape[0] > (4000**2):
                #     if mean_scaling:
                #         scaler = RobustScaler(quantile_range=self.quantile_range)
                #         img_mean = np.mean(img, axis=0)
                #         scaler.fit(img_mean.reshape(-1, 1)).reshape(img.shape)
                #         for i, band in enumerate(img):
                #             img[i] = scaler.transform(band.reshape(-1, 1)).reshape(band.shape)
                #             img[i] = ((img[i] - img_mean.min()) / (img_mean.max() - img_mean.min()))
                #     else:
                #         for i, band in enumerate(img):
                #             scaler = RobustScaler(quantile_range=self.quantile_range)
                #             img[i] = scaler.fit_transform(band.reshape(-1, 1)).reshape(band.shape)
                #             img[i] = ((img[i] - band.min()) / (band.max() - band.min()))
                # else:
                #     # Do a simple max scaling
                #     img = img.astype(np.float32)
                #     for i, band in enumerate(img):
                #         img[i] /= band.max()
                if mean_scaling:
                    scaler = RobustScaler(quantile_range=self.scalers['Mean'].quantile_range)
                    img_mean = np.mean(img, axis=0)
                    scaler.fit(img_mean.reshape(-1, 1))
                    for i, band in enumerate(img):
                        img[i] = scaler.transform(band.reshape(-1, 1)).reshape(band.shape)
                        img[i] = ((img[i] - img_mean.min()) / (img_mean.max() - img_mean.min()))
                else:
                    for i, band in enumerate(img):
                        scaler = RobustScaler(quantile_range=self.scalers[['Red', 'Green', 'Blue'][i]].quantile_range)
                        img[i] = scaler.fit_transform(band.reshape(-1, 1)).reshape(band.shape)
                        img[i] = ((img[i] - band.min()) / (band.max() - band.min()))
            # Put channel as last dimension
            img = np.moveaxis(img, 0, -1)
        return img


class SatTilesScaler(Scaler):
    """
        Simple class to hold all the methods to normalize satellite tiles
    """

    def normalize_band(self, img: np.array, method: str = 'max_scaling') -> np.array:
        """
        Normalize a single band of a satellite product tile

        :param img: np.array, the band to normalize
        :param method: str, the method to use for normalizing the band
        """
        if method == 'absolute_scaling':
            dtype = img.dtype
            img = img.astype(float)
            img /= np.iinfo(dtype).max()
            # if dtype == np.uint8:
            #     img /= 255
            # else:
            #     # We simply move the statistics higher in the range, and then normalize between 0 and 1
            #     img *= 100
            #     img /= np.iinfo(dtype).max()
        elif method == 'max_scaling':
            img /= img.max()
        elif method == 'min_max_scaling':
            img = (img-img.min())/(img.max()-img.min())
        elif method == 'uniform_scaling':
            img = QuantileTransformer(output_distribution='uniform',
                                      random_state=42).fit_transform(img.reshape(-1, 1)).reshape(img.shape)
        return img

    def normalize_product(self, img: np.array, method: str = 'max_scaling', mean_scaling: bool = True,) -> np.array:
        """
        Normalize all bands of a satellite product tile

        :param img: np.array, the satellite product to normalize
        :param method: str, strategy for normalization
        :param mean_scaling: bool, whether to use the statistics from the mean intensity band
        """
        # Find the image channel dimension
        min_dim = np.argmin(img.shape)
        if img.shape[min_dim] > 3:
            raise RuntimeError('Normalization works only for RGB images, sorry!')
        else:
            # Convert to float but retain original datatype
            img_dtype = img.dtype
            img = img.astype(np.float32)
            # Move the channel as the first dimension
            img = np.moveaxis(img, min_dim, 0)
            if method == 'absolute_scaling':
                for i, band in enumerate(['Red', 'Green', 'Blue']):
                    img[i] /= np.iinfo(img_dtype).max
                    # if img_dtype == np.uint8:
                    #     img[i] /= 255
                    # else:
                    #     # We simply stretch the statistics up higher in the dynamic range, and then normalize
                    #     img[i] *= 100  # magic multiplicative constant by Paolo
                    #     img[i] /= np.iinfo(img_dtype).max
            elif method == 'max_scaling':
                mean_max = np.mean(img, axis=0).max()
                for i, band in enumerate(['Red', 'Green', 'Blue']):
                    if mean_scaling:
                        img[i] /= mean_max
                    else:
                        img[i] /= img[i].max()
            elif method == 'min_max_scaling':
                min_prod, max_prod = np.mean(img, axis=0).min(), np.mean(img, axis=0).max()
                for i, band in enumerate(['Red', 'Green', 'Blue']):
                    if mean_scaling:
                        img[i] = (img[i]-min_prod)/(max_prod-min_prod)
                    else:
                        min_prod, max_prod = img[i].min(), img[i].max()
                        img[i] = (img[i]-min_prod)/(max_prod-min_prod)
            elif method == 'uniform_scaling':
                if mean_scaling:  # bring the condition check one outer level for not computing every time the stats
                    transform = QuantileTransformer(output_distribution='uniform',
                                                    random_state=42).fit(np.mean(img, axis=0).reshape(-1, 1))
                    img = transform.transform(img.reshape(-1, 1)).reshape(img.shape)
                else:
                    for i, band in enumerate(['Red', 'Green', 'Blue']):
                        img[i] = QuantileTransformer(output_distribution='uniform',
                                                     random_state=42).fit_transform(img[i].reshape(-1, 1)).reshape(img[i].shape)
            # Move the channel as the last dimension
            img = np.moveaxis(img, 0, -1)
        return img

    def open_and_normalize_product(self, path: str, method: str = 'max_scaling', mean_scaling: bool = True) -> np.array:
        """
        As the name suggests
        """
        # Open the product
        with rasterio.open(path, 'r') as src:
            img = src.read()
        # Normalize it
        img = self.normalize_product(img, method, mean_scaling)
        return img


class DBLDataGenerator(Sequence):
    """
    DBLDataGenerator
    Class used for the batches generation and organization in Keras for a training employing the DBL loss.
    """

    def __init__(self, batch_size, patch_size, data_dir, num_iteration, num_pos_pertile, num_tiles_peracq,
                 batch_num_tiles_peracq, split_seed: int = 42, scaler_type: str = '99th_percentile',
                 mean_scaling_strategy: bool = True, input_norm: str = 'max_abs', he_norm: str = None,
                 gray_scale: bool = False, p_aug: float = 0.0):
        """
        :param batch_size: size of batches, corresponds to the number of different acquisitions considered in the batch
        :param patch_size: size of patches extracted from each tile of the acquisitions
        :param data_dir: directory containing the data to load.
            Data will be loaded as list of matrices, where each matrix contains a list of tiles for an RGB product.
            Each element of the list_acq is therefore itself a list, and the elements of this nested lists are the tiles
            for that acquisition, so matrices in the form 3 x Nrows x Ncols.
            This data will be saved in a member variable of the class.
        :param num_iteration: int, number of iterations defining an epoch
        :param num_pos_pertile: int, number of random positions for the extraction of the patches from the tiles
        :param num_tiles_peracq: int, number of tiles considered from each acquisition
        :param batch_num_tiles_peracq: int, number of tiles considered from each acquisition inside each batch
        :param seed: int, random seed used for selecting the tiles from each acquisition
        :param scaler_type: str, Quantile used for scaling the data
        :param mean_scaling_strategy: bool, whether to use "mean" scaling or not
        :param input_norm: str, strategy for input norm of the tiles scaler
        :param he_norm: str, strategy for histogram equalization for tiles scaler
        :param p_aug: flaot, probability of applying a random contrast augmentation
        """

        # Set members
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_position_pertile = num_pos_pertile
        self.num_tiles_peracq = num_tiles_peracq
        self.batch_num_tiles_peracq = batch_num_tiles_peracq
        self.num_patch_perbatch = batch_size * self.num_position_pertile * self.batch_num_tiles_peracq
        self.patch_size = patch_size
        self.num_iteration = num_iteration
        self.mean_scaling_strategy = mean_scaling_strategy
        self.input_norm = input_norm
        self.he_norm = he_norm
        self.gray_scale = gray_scale
        self.p_aug = p_aug
        if p_aug > 0:
            # Let's keep contrast augmentation aside for a moment
            # self.augs = iaa.Sometimes(p_aug, iaa.OneOf([iaa.SigmoidContrast(gain=(5, 20), cutoff=(0.25, 0.75)),
            #                                             iaa.LogContrast(gain=(0.6, 1.4)),
            #                                             iaa.LinearContrast((0.4, 1.6))]))
            # Let's tru adding uniform equalization
            # self.augs = iaa.Sometimes(p_aug,
            #                           iaa.Lambda(lambda x, random_state, parents, hooks:
            #                                      [SatTilesScaler().normalize_product(image, 'uniform_scaling',
            #                                                                         self.mean_scaling_strategy)
            #                                       for image in x],
            #                                      lambda x, random_state, parents, hooks: x))
            self.augs = iaa.Sometimes(p_aug,
                                      iaa.Lambda(lambda x, random_state, parents, hooks:
                                                 [skimage.exposure.equalize_hist(image, nbins=50)
                                                  for image in x],
                                                 lambda x, random_state, parents, hooks: x))

        # Load the data (first, set the random split for loading the patches)
        self.split_seed = np.random.random_integers(42) if split_seed is None else split_seed
        self.randomState = np.random.RandomState(self.split_seed)
        self.list_data = make_dir_split(self.data_dir, self.num_tiles_peracq, seed=self.split_seed)
        self.list_acq = []
        for product, img_list in self.list_data.items():
            # Load the Scaler
            if scaler_type == '99th_percentile':
                self.scaler = load(os.path.join(self.data_dir, product, '99_SatProductRobustScaler.joblib'))
                # Load and normalize the data
                self.list_acq.append(
                    [self.scaler.open_and_normalize_product(path, mean_scaling_strategy) for path in img_list])
            elif scaler_type == '95th_percentile':
                self.scaler = load(os.path.join(self.data_dir, product, '95_SatProductRobustScaler.joblib'))
                # Load and normalize the data
                self.list_acq.append(
                    [self.scaler.open_and_normalize_product(path, mean_scaling_strategy) for path in img_list])
            elif scaler_type == 'sat_max':
                self.scaler = load(os.path.join(self.data_dir, product, 'SatProductMaxScaler.joblib'))
                # Load and normalize the data
                self.list_acq.append(
                    [self.scaler.open_and_normalize_product(path, mean_scaling_strategy) for path in img_list])
            elif scaler_type == 'sat_tiles_scaler':
                self.scaler = SatTilesScaler()
                # Load and normalize the data
                self.list_acq.append(
                    [self.scaler.open_and_normalize_product(path, input_norm, mean_scaling_strategy) for path in img_list])

        # Set the last members
        self.num_acq = len(self.list_acq)
        self.num_batches = self.num_acq // self.batch_size
        assert ((self.num_acq % self.batch_size) == 0)  # the number of acquisitions must be a multiple of the batch size

        # Reset the random seed
        if split_seed is None:
            self.split_seed = np.random.random_integers(42)
        self.randomState = np.random.RandomState(self.split_seed)

        # Select data for iteration
        self._indices0 = np.arange(0, len(self.list_acq))

        # Initialize random permutation
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        # return self.num_batches
        return self.num_iteration

    def on_epoch_end(self):
        """
        Shuffle indices after one epoch
        """
        self.randomState.shuffle(self._indices0)

    def print_info(self):
        print(self.batch_size, self.num_batches)

    def __getitem__(self, index):
        """
        Generate one batch of data with relative labels
        """

        self.randomState.shuffle(self._indices0)

        X_batch = []
        Y_batch = np.zeros((self.num_patch_perbatch, self.num_patch_perbatch))
        indexes = self._indices0[:self.batch_size]

        # loop over acquisitions
        patch_idx = 0
        for i, idx_d in enumerate(indexes):

            img_indexes = np.random.permutation(len(self.list_acq[idx_d]))[:self.batch_num_tiles_peracq]
            p_y = np.random.permutation(self.list_acq[idx_d][0].shape[0] - self.patch_size - 1)[:self.num_position_pertile]
            p_x = np.random.permutation(self.list_acq[idx_d][0].shape[1] - self.patch_size - 1)[:self.num_position_pertile]

            for pos_idx in range(len(p_x)):
                # load num_pos*num_tiles_peracq patch per device
                for n_i in range(self.batch_num_tiles_peracq):
                    if self.p_aug > 0:
                        batch_sample = self.list_acq[idx_d][img_indexes[n_i]]
                        # if batch_sample.min() < 0:
                        #     batch_sample = (batch_sample-batch_sample.min())/(batch_sample.max()-batch_sample.min())
                        batch_sample = self.augs(image=batch_sample)
                    else:
                        batch_sample = self.list_acq[idx_d][img_indexes[n_i]]
                    X_batch.append(batch_sample[
                                   p_y[pos_idx]:p_y[pos_idx] + self.patch_size,
                                   p_x[pos_idx]:p_x[pos_idx] + self.patch_size])

            # Prepare the matrix of labels
            Y_batch[patch_idx:patch_idx + self.batch_num_tiles_peracq * self.num_position_pertile,
                    patch_idx:patch_idx + self.batch_num_tiles_peracq * self.num_position_pertile] = 1.
            patch_idx += self.batch_num_tiles_peracq * self.num_position_pertile

        # Convert to numpy arrays
        X_batch = np.asarray(X_batch)
        #X_batch = np.expand_dims(X_batch, -1)

        return X_batch, Y_batch


class DBLDataGeneratorFixedPos(Sequence):
    """
    DBLDataGenerator
    Class used for the batches generation and organization in Keras for a training employing the DBL loss.
    FixedPos -> we enforce the label also on the extraction position of the different patches
    """

    def __init__(self, batch_size, patch_size, data_dir, num_iteration, num_pos_pertile, num_tiles_peracq,
                 batch_num_tiles_peracq, split_seed: int = 42, scaler_type: str = '99th_percentile',
                 mean_scaling_strategy: bool = True, input_norm: str = 'max_abs', he_norm: str = None,
                 gray_scale: bool = False, p_aug: float = 0.0):
        """
        :param batch_size: size of batches, corresponds to the number of different acquisitions considered in the batch
        :param patch_size: size of patches extracted from each tile of the acquisitions
        :param data_dir: directory containing the data to load.
            Data will be loaded as list of matrices, where each matrix contains a list of tiles for an RGB product.
            Each element of the list_acq is therefore itself a list, and the elements of this nested lists are the tiles
            for that acquisition, so matrices in the form 3 x Nrows x Ncols.
            This data will be saved in a member variable of the class.
        :param num_iteration: int, number of iterations defining an epoch
        :param num_pos_pertile: int, number of random positions for the extraction of the patches from the tiles
        :param num_tiles_peracq: int, number of tiles considered from each acquisition
        :param batch_num_tiles_peracq: int, number of tiles considered from each acquisition inside each batch
        :param seed: int, random seed used for selecting the tiles from each acquisition
        :param scaler_type: str, Quantile used for scaling the data
        :param mean_scaling_strategy: bool, whether to use "mean" scaling or not
        :param input_norm: str, strategy for input norm (mutually exclusive with mean_scaling_strategy)
        :param he_norm: str, strategy for histogram equalization (mutually exclusive with mean_scaling_strategy)
        :param gray_scale: bool, whether to convert the RGB bands in mean intensity (mutually exclusive with mean_scaling_strategy)
        :param p_aug: flaot, probability of applying a random contrast augmentation
        """

        # Set members
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_position_pertile = num_pos_pertile
        self.num_tiles_peracq = num_tiles_peracq
        self.batch_num_tiles_peracq = batch_num_tiles_peracq
        self.num_patch_perbatch = batch_size * self.num_position_pertile * self.batch_num_tiles_peracq
        self.patch_size = patch_size
        self.num_iteration = num_iteration
        self.mean_scaling_strategy = mean_scaling_strategy
        self.input_norm = input_norm
        self.he_norm = he_norm
        self.gray_scale = gray_scale
        self.p_aug = p_aug
        if p_aug > 0:
            # Let's keep contrast augmentation aside for a moment
            # self.augs = iaa.Sometimes(p_aug, iaa.OneOf([iaa.SigmoidContrast(gain=(5, 20), cutoff=(0.25, 0.75)),
            #                                             iaa.LogContrast(gain=(0.6, 1.4)),
            #                                             iaa.LinearContrast((0.4, 1.6))]))
            # Let's tru adding uniform equalization
            self.augs = iaa.Sometimes(p_aug, iaa.Lambda(
                lambda x: SatTilesScaler().normalize_product(x, 'uniform_scaling', self.mean_scaling_strategy),
                None))

        # Load the data (first, set the random split for loading the patches)
        self.split_seed = np.random.random_integers(42) if split_seed is None else split_seed
        self.randomState = np.random.RandomState(self.split_seed)
        self.list_data = make_dir_split(self.data_dir, self.num_tiles_peracq, seed=self.split_seed)
        self.list_acq = []
        for product, img_list in self.list_data.items():
            # Load the Scaler
            if scaler_type == '99th_percentile':
                self.scaler = load(os.path.join(self.data_dir, product, '99_SatProductRobustScaler.joblib'))
                # Load and normalize the data
                self.list_acq.append(
                    [self.scaler.open_and_normalize_product(path, mean_scaling_strategy) for path in img_list])
            elif scaler_type == '95th_percentile':
                self.scaler = load(os.path.join(self.data_dir, product, '95_SatProductRobustScaler.joblib'))
                # Load and normalize the data
                self.list_acq.append(
                    [self.scaler.open_and_normalize_product(path, mean_scaling_strategy) for path in img_list])
            elif scaler_type == 'sat_max':
                self.scaler = load(os.path.join(self.data_dir, product, 'SatProductMaxScaler.joblib'))
                # Load and normalize the data
                self.list_acq.append(
                    [self.scaler.open_and_normalize_product(path, mean_scaling_strategy) for path in img_list])
            elif scaler_type == 'sat_tiles_scaler':
                self.scaler = SatTilesScaler()
                # Load and normalize the data
                self.list_acq.append(
                    [self.scaler.open_and_normalize_product(path, input_norm, mean_scaling_strategy) for path in
                     img_list])

        # Set the last members
        self.num_acq = len(self.list_acq)
        self.num_batches = self.num_acq // self.batch_size
        assert ((self.num_acq % self.batch_size) == 0)  # the number of acquisitions must be a multiple of the batch size

        # Reset the random seed
        if split_seed is None:
            self.split_seed = np.random.random_integers(42)
        self.randomState = np.random.RandomState(self.split_seed)

        # Select data for iteration
        self._indices0 = np.arange(0, len(self.list_acq))

        # Initialize random permutation
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        # return self.num_batches
        return self.num_iteration

    def on_epoch_end(self):
        """
        Shuffle indices after one epoch
        """
        self.randomState.shuffle(self._indices0)

    def print_info(self):
        print(self.batch_size, self.num_batches)

    def __getitem__(self, index):
        """
        Generate one batch of data with relative labels
        """
        # assert (index < self.num_batches)
        # offset = self.batch_size * index

        self.randomState.shuffle(self._indices0)

        X_batch = []
        Y_batch = np.zeros((self.num_patch_perbatch, self.num_patch_perbatch))
        # indexes = self._indices0[index * self.batch_size: (index + 1)*self.batch_size]
        indexes = self._indices0[:self.batch_size]
        # sel_devices = self.list_data0[indexes]

        # loop over devices
        patch_idx = 0
        for i, idx_d in enumerate(indexes):

            img_indexes = np.random.permutation(len(self.list_acq[idx_d]))[:self.batch_num_tiles_peracq]
            p_y = np.random.permutation(self.list_acq[idx_d][0].shape[0] - self.patch_size - 1)[:self.num_position_pertile]
            p_x = np.random.permutation(self.list_acq[idx_d][0].shape[1] - self.patch_size - 1)[:self.num_position_pertile]

            for pos_idx in range(len(p_x)):
                # Loading batch_num_tiles_peracq patches for acquisition
                for n_i in range(self.batch_num_tiles_peracq):
                    if self.p_aug > 0:
                        batch_sample = self.augs(image=self.list_acq[idx_d][img_indexes[n_i]])
                    else:
                        batch_sample = self.list_acq[idx_d][img_indexes[n_i]]
                    X_batch.append(batch_sample[p_y[pos_idx]:p_y[pos_idx] + self.patch_size,
                                                p_x[pos_idx]:p_x[pos_idx] + self.patch_size])
                Y_batch[patch_idx:patch_idx + self.batch_num_tiles_peracq, patch_idx:patch_idx + self.batch_num_tiles_peracq] = 1.
                patch_idx += self.batch_num_tiles_peracq


        X_batch = np.asarray(X_batch)
        #X_batch = np.expand_dims(X_batch, -1)

        return X_batch, Y_batch

# Helpers functions #

def load_and_normalize(path: str, use_he: bool = False) -> np.array:
    """
    Load a generic image and normalize it as a float32 between -1 and 1
    following a normal distribution
    :param path: str, the path of the image
    :param use_he: bool, whether to perform histogram equalization following a normal distribution
    :return: np.array float32 of the normalized image
    """
    # Find extension first
    extension = ntpath.split(path)[-1].split('.')[-1]

    # Load the image according to the file type
    if extension == 'tiff':
        with rasterio.open(path) as src:
            img = src.read()
    elif extension == 'png':
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    elif extension == 'npy':
        img = np.load(path)

    if use_he:
        # Histogram equalization
        img = QuantileTransformer(output_distribution='normal',
                                  random_state=42).fit_transform(img.reshape(-1, 1)).reshape(img.shape).astype(np.float32)
    else:
        # Perform a simple multiplicative scaling (used for the "standard" detectors)
        if img.dtype == np.uint8:
            img = img.astype(np.float32)
            img /= 255
        elif img.dtype == np.uint16:
            img = img.astype(np.float32)
            img /= (2 ** 16 - 1)
            img *= 100
            img = np.clip(img, 0, 1).astype(np.float32)
        else:
            img = img.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min())

    return np.squeeze(img)


def load_and_normalize_RGB(path: str, gray_scale: bool = False,
                           norm: str='max_abs', use_he: str = None) -> np.array:
    """
    Load a generic RGB image and normalize it as a float32 in grayscale
    :param path: str, the path of the image
    :param norm: str, normalization of the input (either 'max_abs' -> 0, 1 or 'min_max' -> -1, 1)
    :param use_he: str, whether to perform histogram equalization following the indicated distribution
    :return: np.array float32 of the normalized image
    """

    # Load
    ext = os.path.split(path)[-1].split('.')[-1]
    if ext == 'tiff':
        with rasterio.open(path, 'r') as src:
            img = src.read()
    else:
        img = np.array(Image.open(path))
    img_float = np.squeeze(img).astype(np.float32)
    # Normalize
    if gray_scale:
        channel_idx = np.argmin(img.shape)  # move the color channel first
        img_float = np.moveaxis(img_float, channel_idx, 0)
        img_float = (0.299 * img_float[0, :, :] + 0.587 * img_float[1, :, :] + 0.114 * img_float[2, :, :]).astype(np.float32)  # from Unina
        # THIS IS NOT CORRECT, RGB images are not provided in the luminance chrominance space!

    if norm == 'max_abs':
        img_float /= np.float32(np.iinfo(img.dtype).max)
    elif norm == 'min_max':
        img_float -= np.float32(np.iinfo(img.dtype).max)/2
        img_float /= np.float32(np.iinfo(img.dtype).max)/2

    if use_he:
        # Histogram equalization
        img_float = QuantileTransformer(output_distribution=use_he,
                                        random_state=42).fit_transform(img_float.reshape(-1, 1)).reshape(img_float.shape).astype(np.float32)

    # Move the channel as the last dimension
    min_dim = np.argmin(img_float.shape)
    img_float = np.moveaxis(img_float, min_dim, -1)

    return img_float


def normalize(img: np.array, mod: str) -> np.array:
    """
    Normalize image according to different HE and range normalizations
    :param img: np.ndarray
    :param mod: str, selects the modality for range normalization
    :return: np.ndarray, the normalized image
    """
    if mod == 'norm_he':
        # Histogram equalization
        img = QuantileTransformer(output_distribution='normal').fit_transform(img.reshape(-1, 1)).reshape(
            img.shape).astype(np.float32)
    if mod == 'norm_0_mean':
        # Histogram equalization
        img = QuantileTransformer(output_distribution='normal').fit_transform(img.reshape(-1, 1)).reshape(
            img.shape).astype(np.float32)
        # -1:1 normalization
        img /= img.max()
    elif mod == 'norm_0.5_mean':
        # Histogram equalization
        img = QuantileTransformer(output_distribution='normal').fit_transform(img.reshape(-1, 1)).reshape(
            img.shape).astype(np.float32)
        # 0:1 normalization
        img = (img-img.min())/(img.max()-img.min())
    elif mod == 'raw_scaling':
        img = img.astype(np.float32)
        img /= (2**16-1)
    elif mod == 'mul_scaling':
        if img.dtype == np.uint8:
            img = img.astype(np.float32)
            img /= 255
        elif img.dtype == np.uint16:
            img = img.astype(np.float32)
            img /= (2 ** 16 - 1)
            img *= 100
            img = np.clip(img, 0, 1).astype(np.float32)
        else:
            img = img.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min())
    return np.squeeze(img)


# --- MAIN FUNCTION for debugging purposes --- #
if __name__ == '__main__':
    from functools import partial
    import matplotlib.pyplot as plt

    train_dir = '/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/patches/train_patches'
    val_dir = '/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/patches/val_patches'

    # --- Load list of images for training and validation --- #
    list_train = make_dir_split(train_dir, num_tiles_peracq=200,
                                seed=42)
    list_valid = make_dir_split(val_dir, num_tiles_peracq=200, seed=42)

    train_data = [list(map(partial(load_and_normalize_RGB, norm='max_abs', use_he='uniform'), imgs_list))
                  for imgs_list in list_train.values()]
    val_data = [list(map(partial(load_and_normalize_RGB, norm='max_abs', use_he='uniform'), imgs_list)) for imgs_list in
                list_valid.values()]

    # DO THE SAME with the DBLDataGenerator
    train_data_generator = DBLDataGenerator(batch_size=5, patch_size=48, data_dir=train_dir,
                                            num_iteration=128, num_pos_pertile=6,
                                            num_tiles_peracq=200)
    valid_data_generator = DBLDataGenerator(batch_size=5, patch_size=48, data_dir=val_dir,
                                            num_iteration=128, num_pos_pertile=6,
                                            num_tiles_peracq=200)
    # Check that everything is fine
    print(f'Train paths are extracted correctly? '
          f'{np.sum([elem==list_train[idx] for idx, elem in train_data_generator.list_data.items()])==len(list_train)}')
    print(f'Val paths are extracted correctly? '
          f'{np.sum([elem == list_valid[idx] for idx, elem in valid_data_generator.list_data.items()]) == len(list_valid)}')
    print(f'Train data is processed correctly? {len(train_data)==len(train_data_generator.list_acq) and np.sum([len(elem) == len(train_data[idx]) for idx, elem in enumerate(train_data_generator.list_acq)]) == len(train_data)}')
    print(f'Val data is processed correctly? {len(val_data) == len(valid_data_generator.list_acq) and np.sum([len(elem) == len(val_data[idx]) for idx, elem in enumerate(valid_data_generator.list_acq)]) == len(val_data)}')

    # Let's plot some samples for fun
    for img_list in train_data_generator.list_acq:
        fig, axs = plt.subplots(2, 5, figsize=(24, 12))
        for i in range(5):
            axs[0][i].imshow(np.moveaxis(img_list[i], 0, 2))
            axs[0][i].axis('off')
            colors = ['r', 'g', 'b']
            labels = ['Red', 'Green', 'Blue']
            for j, band in enumerate(img_list[i]):
                axs[1][i].hist(band.flatten(), bins=200, label=labels[j],
                                 color=colors[j])
                axs[1][i].legend()
        plt.show()
