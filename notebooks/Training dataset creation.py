#!/usr/bin/env python
# coding: utf-8

# # Training dataset creation
# A small notebook for creating the training dataset for the RGB fingerprint extractor.

# ## Libraries import

# In[2]:


import rasterio
import numpy as np
import os
import ntpath
import glob
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('..')
from isplutils.PatchExtractor import PatchExtractor
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Times']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


# ## Helpers functions

# In[4]:


"""
Some image functions we are trying for normalizing 16-bit GRD images and convert them to 8-bits images.
For each scaling we provide the 8-bit conversion process too.

Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
"""
from sklearn.preprocessing import RobustScaler, QuantileTransformer


## Normalization functions

def raw_scaling(img: np.ndarray, conversion_8: bool = False) -> np.ndarray:
    """
    Standard min-max scaler

    :param img: np.ndarray, input image (tested GRD 16-bits)
    :param conversion_8: bool, wheter to convert to 8-bits depth
    :return: np.ndarray, normalized numpy array
    """
    norm = ((img-img.min())/(img.max()-img.min()))
    if conversion_8: # here
        norm = (norm*255).astype(np.uint8)
    return norm


def db_scaling(img: np.ndarray, conversion_8: bool = False) -> np.ndarray:
	"""
	Puts the image in logarithmic scale with a factor 10 
	
	:param img: np.ndarray, input image (tested GRD 16-bits)
	:param conversion_8: bool, wheter to convert to 8-bits depth
	:return: np.ndarray, normalized numpy array
	"""
	norm = np.log10(img)
	if conversion_8:
		norm = (norm-norm.min())/(norm.max()-norm.min())  # np.log10 provides the image already in float32, however the range may not be restricted 0-1
		norm = (norm*255).astype(np.uint8)
	return norm


def mul_scaling(img: np.ndarray,  mul_factor: int = 100, conversion_8: bool = False) -> np.ndarray:
	"""
	Simple scaling of the dynamic with a multiplicative factor
	
	:param img: np.ndarray, input image (tested GRD 16-bits)
	:param mul_factor: int, multiplicative factor for the scaling
	:param conversion_8: bool, wheter to convert to 8-bits depth
	:return: np.ndarray, normalized numpy array
	"""
	norm = ((img-img.min())/(img.max()-img.min()))
	norm *= mul_factor
	if conversion_8:
		norm = (norm*255).astype(np.uint8)
	return norm


def janos_scaling(img: np.ndarray, conversion_8: bool = False) -> np.ndarray:
	"""
	Nice dynamic scaling using exponential functions, original author Janos Horvath - horvath5@purdue.edu
	
	:param img: np.ndarray, input image (tested GRD 16-bits)
	:param conversion_8: bool, wheter to convert to 8-bits depth
	:return: np.ndarray, normalized numpy array
	"""
	norm = 2/((1+np.exp(-(img*40/256/256))))-1
	if conversion_8:
		norm = (norm*255).astype(np.uint8)
	return norm


def mz_scaling(img: np.ndarray, conversion_8: bool = False) -> np.ndarray:
    """
    Modified Z-score normalization taken from Sun et al, “DeepInSAR—A Deep Learning Framework for SAR Interferometric Phase Restoration and Coherence Estimation”

    :param img: np.ndarray, input image (tested GRD 16-bits)
    :param conversion_8: bool, wheter to convert to 8-bits depth
    :return: np.ndarray, normalized numpy array
    """
    shape = img.shape
    img = img.flatten()
    img_med = np.median(img)
    mad = np.median(np.abs(img - np.median(img)))
    #mad=0.1656
    mz = 0.6745 * ((img - np.median(img)) / mad)  # 0.6475 is the 75th quartile of the normal distribution, removes the outlier
    #mz = 0.6745 * ((a - 0.7097678) / mad)
    mz = (np.tanh(mz / 7) + 1) / 2
    mz_min = mz.min()
    mz_max = mz.max()
    mz = (mz - mz.min()) / (mz.max() - mz.min())
    norm = mz.reshape(shape)
    if conversion_8:
        norm = (norm*255).astype(np.uint8)
    return norm


## Range equalizers

def robust_scaling(img: np.ndarray) -> np.ndarray:
	"""
	RobustScaler: eliminates outliers, reproject the images in the same range
	
	:param img: np.ndarray, input image (tested GRD 16-bits)
	:return: np.ndarray, normalized numpy array
	"""
	return RobustScaler().fit_transform(img.reshape(-1, 1)).reshape(img.shape)
	

def quantile_scaling(img: np.ndarray, dist: str) -> np.ndarray:
	"""
	QuantileTransformer: maps the image range to a specific stastical distribution
	
	:param img: np.ndarray, input image (tested GRD 16-bits)
	:param distribution: str, distribution for mapping the image (either 'uniform' or 'normal')
	:return: np.ndarray, normalized numpy array
	"""
	return QuantileTransformer(output_distribution=dist).fit_transform(img.reshape(-1, 1)).reshape(img.shape)

def extract_patches(raster, mask, dim=(3, 256, 256)):
    # Let's instantiate the PatchExtractor
    pe = PatchExtractor(dim=dim)
    pe_mask = PatchExtractor(dim=dim)
    # Let's extract patches for both the images and masks
    img_patches = pe.extract(raster).reshape((-1,)+dim)
    mask_patches = pe_mask.extract(mask.astype(bool)).reshape((-1,)+dim)
    # Let's find patches with no data elements
    nodata_flags = np.array([np.sum(mask_patch)!=np.prod(dim) for mask_patch in mask_patches])
    # Let's pick only patches without no data elements
    img_patches = img_patches[~nodata_flags]
    return img_patches


# ## Let's collect all the info on these products

# In[5]:


# INPE images
bands = glob.glob('/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/full_res_products/**/*.tif*', recursive=True)
bands = [band for band in bands if ('analytic_sr' not in band) and ('patches' not in band)]
data_df = pd.concat([pd.concat({acq.split('/')[8]: pd.DataFrame(index=[acq])}, names=['Original_product', 'Bands path']) for acq in bands])
# Let's pick up the sensors first and bands first
data_df['sensor'] = ''
data_df['bands_num'] = ''
for i, r in data_df.iterrows():
    data_df.loc[i, 'sensor'] = i[0].split('_')[2]
    if 'AMAZONIA' in i[0]:
        data_df.loc[i, 'sensor'] = 'AWFI'
    if 'L4_' in i[1]:
        data_df.loc[i, 'bands_num'] = i[1].split('L4_')[1].split('.')[0]
    else:
        data_df.loc[i, 'bands_num'] = i[1].split('L2_')[1].split('.')[0]
# Landsat bands
bands = glob.glob('/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/full_res_products/**/*.TIF', recursive=True)
tmp_df = pd.concat([pd.concat({acq.split('/')[8]: pd.DataFrame(index=[acq])}, names=['Original_product', 'Bands path']) for acq in bands])
tmp_df['sensor'] = 'OLI'
data_df = pd.concat([data_df, tmp_df])
# Let's pick up the sensors first and bands first
for i, r in tmp_df.iterrows():
    data_df.loc[i, 'bands_num'] = i[1].split('/')[-1].split('_')[1].split('.')[0]
# Sentinel samples
bands = glob.glob('/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/full_res_products/**/IMG_DATA/R10m/*B0*.jp2', recursive=True)
tmp_df = pd.concat([pd.concat({acq.split('/')[8]: pd.DataFrame(index=[acq])}, names=['Original_product', 'Bands path']) for acq in bands])
data_df = pd.concat([data_df, tmp_df])
# Let's pick up the sensors first and bands first
for i, r in tmp_df.iterrows():
    data_df.loc[i, 'bands_num'] = i[1].split('/')[-1].split('_')[-2]
    data_df.loc[i, 'sensor'] = i[0].split('_')[0]
# PlanetLab samples
bands = glob.glob('/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/full_res_products/**/*.tif', recursive=True)
bands = [band for band in bands if ('analytic' in band) and ('udm.tif' not in band) and ('udm2.tif' not in band)]
tmp_df = pd.concat([pd.concat({acq.split('/')[8]: pd.DataFrame(index=[acq])}, names=['Original_product', 'Bands path']) for acq in bands])
data_df = pd.concat([data_df, tmp_df])
data_df


# ### Add height and width

# In[6]:


# We can add some information in between too
for product in data_df.index.get_level_values(0).unique():
    # Select the product
    df = data_df.loc[product]
    fsize = 20
    for idx, band in enumerate(df.index.get_level_values(0).unique()):
        # Load the band
        with rasterio.open(band, 'r') as src:
            # Add info
            data_df.loc[(product, band), 'height'] = src.profile['height']
            data_df.loc[(product, band), 'width'] = src.profile['width']
            data_df.loc[(product, band), 'bit_depth'] = src.profile['dtype']
data_df


# ### Add sensor info

# In[7]:


# Let's add a more comprehensible definition for bands
bands_dict = {sensor: dict() for sensor in data_df['sensor'].unique()}
bands_dict['PAN10M'] = {'BAND3': 'Red', 'BAND4': 'NIR', 'BAND2': 'Green'}
bands_dict['MUX'] = {'BAND5': 'Blue', 'BAND6': 'Green', 'BAND7': 'Red', 'BAND8': 'NIR'}
bands_dict['WFI'] = {'BAND13': 'Blue', 'BAND14': 'Green', 'BAND15': 'Red', 'BAND16': 'NIR'}
bands_dict['CCD1XS'] = {'BAND1': 'Blue', 'BAND2': 'Green', 'BAND3': 'Red', 'BAND4': 'NIR', 'BAND5': 'PAN'}
bands_dict['AWFI'] = {'BAND1': 'Blue', 'BAND2': 'Green', 'BAND3': 'Red', 'BAND4': 'NIR'}
bands_dict['OLI'] = {'B1': 'Coastal aerosol', 'B2': 'Blue', 'B3': 'Green', 'B4': 'Red'}
bands_dict['S2A'] = {'B02': 'Blue', 'B03': 'Green', 'B04': 'Red', 'B08': 'NIR'}
bands_dict['S2B'] = {'B02': 'Blue', 'B03': 'Green', 'B04': 'Red', 'B08': 'NIR'}
no_pl_df = data_df.drop(['PlanetScope_Campinas_psorthotile_analytic_sr_udm2',
                         'RapidEye2_Santos_reorthotile_analytic_sr',
                         'RapidEye4_Sao_Miguel_Arcangeo_reorthotile_analytic_sr'])
for i in no_pl_df.index:
    data_df.loc[i, 'bands_name'] = bands_dict[no_pl_df.loc[i, 'sensor']][no_pl_df.loc[i, 'bands_num']]
data_df.loc['PlanetScope_Campinas_psorthotile_analytic_sr_udm2', 'sensor'] = 'PSB.SD'
data_df.loc['PlanetScope_Campinas_psorthotile_analytic_sr_udm2', 'bands_name'] = 'RGBN'
data_df.loc['RapidEye2_Santos_reorthotile_analytic_sr', 'sensor'] = 'JSS 56'
data_df.loc['RapidEye2_Santos_reorthotile_analytic_sr', 'bands_name'] = 'RGBReN'
data_df.loc['RapidEye4_Sao_Miguel_Arcangeo_reorthotile_analytic_sr', 'sensor'] = 'JSS 56'
data_df.loc['RapidEye4_Sao_Miguel_Arcangeo_reorthotile_analytic_sr', 'bands_name'] = 'RGBReN'
data_df


# ### Let's crop some patches out of it

# In[8]:


save_path = '/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/patches'

dim = (3, 256, 256)
fsize = 30
for product in data_df.drop('CBERS_4_PAN10M_20220905_151_116').index.get_level_values(0).unique():
#for product in ['RapidEye4_Sao_Miguel_Arcangeo_reorthotile_analytic_sr', 'RapidEye2_Santos_reorthotile_analytic_sr', 'PlanetScope_Campinas_psorthotile_analytic_sr_udm2']:
    
    # Select the product
    df = data_df.loc[product]
    
    bands = dict()
    masks = dict()
    for i, r in df.iterrows():
        if r['bands_name'] in ['Red', 'Green', 'Blue']:
            # Load the band and mask
            with rasterio.open(i, 'r') as src:
                band = src.read()
                mask = src.read_masks()
                dtype = src.profile['dtype']
                profile = src.profile
            
            bands[r['bands_name']] = band
            # Add check for CBERS2 and Landsat, their masks are not reliable
            if product not in ['CBERS_2_CCD1XS_20081230_156_116', 'CBERS_2B_CCD1XS_20090905_167_120', 'LO82240772021365CUB00']:
                masks[r['bands_name']] = mask
            else:
                masks[r['bands_name']] = (band!=0).astype(np.uint8)  
                
        elif r['bands_name'] in ['RGBN', 'RGBReN']:
            
            # Load bands
            with rasterio.open(i, 'r') as src:
                raster = src.read()
                dtype = src.profile['dtype']
                profile = src.profile
                
            # Add the bands to the bands dictionary
            bands['Red'] = raster[2][np.newaxis, :, :]
            bands['Green'] = raster[1][np.newaxis, :, :]
            bands['Blue'] = raster[0][np.newaxis, :, :]
            
            # Load masks
            if 'RapidEye' in i:
                i = i.replace('Analytic_SR', 'udm')
                with rasterio.open(i, 'r') as src:
                    masks_raster = src.read()
                masks_raster[masks_raster==2] = 0  # fix the cloud pixels
                masks_raster[masks_raster==0] = 255  # revert the mask
                masks_raster[masks_raster!=255] = 0  # change the other pixels to 0
                masks['Red'] = masks['Green'] = masks['Blue'] = masks_raster.copy()
            else:
                i = i.replace('BGRN_SR', 'udm2')
                with rasterio.open(i, 'r') as src:
                    masks_raster = src.read()
                final_mask = masks_raster[0]+masks_raster[2]+masks_raster[5]
                masks['Red'] = masks['Green'] = masks['Blue'] = final_mask[np.newaxis, :, :]
    
    # Plot histograms for patches bands
    patches = extract_patches(raster=np.concatenate([bands['Red'], bands['Green'], bands['Blue']]),
                              mask=np.concatenate([masks['Red'], masks['Green'], masks['Blue']]), dim=dim)
    fig, axs = plt.subplots(1, 4, figsize=(48, 12))
    idxs = np.random.choice(len(patches), 4)
    for i, idx in enumerate(idxs):
        axs[i].imshow(quantile_scaling(np.moveaxis(patches[idx], 0, 2), 'uniform'))  # let's quantize just for visualise stuff
        axs[i].set_title(f'Bit depth {patches[idx].dtype}', fontsize=fsize)
        axs[i].axis('off')
    fig.suptitle(f"{product} bands and patches, {dtype} bits depth", fontsize=fsize+5)
    plt.show()


# ### Let's see some values distributions

# In[15]:


save_path = '/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/patches'

dim = (3, 256, 256)
fsize = 30
for product in data_df.drop('CBERS_4_PAN10M_20220905_151_116').index.get_level_values(0).unique():
#for product in ['RapidEye4_Sao_Miguel_Arcangeo_reorthotile_analytic_sr', 'RapidEye2_Santos_reorthotile_analytic_sr', 'PlanetScope_Campinas_psorthotile_analytic_sr_udm2']:
    
    # Select the product
    df = data_df.loc[product]
    
    bands = dict()
    masks = dict()
    for i, r in df.iterrows():
        if r['bands_name'] in ['Red', 'Green', 'Blue']:
            # Load the band and mask
            with rasterio.open(i, 'r') as src:
                band = src.read()
                mask = src.read_masks()
                dtype = src.profile['dtype']
            
            bands[r['bands_name']] = band
            # Add check for CBERS2 and Landsat, their masks are not reliable
            if product not in ['CBERS_2_CCD1XS_20081230_156_116', 'CBERS_2B_CCD1XS_20090905_167_120', 'LO82240772021365CUB00']:
                masks[r['bands_name']] = mask
            else:
                masks[r['bands_name']] = (band!=0).astype(np.uint8)  
                
        elif r['bands_name'] in ['RGBN', 'RGBReN']:
            
            # Load bands
            with rasterio.open(i, 'r') as src:
                raster = src.read()
                dtype = src.profile['dtype']
            # Add the bands to the bands dictionary
            bands['Red'] = raster[2][np.newaxis, :, :]
            bands['Green'] = raster[1][np.newaxis, :, :]
            bands['Blue'] = raster[0][np.newaxis, :, :]
            
            # Load masks
            if 'RapidEye' in i:
                i = i.replace('Analytic_SR', 'udm')
                with rasterio.open(i, 'r') as src:
                    masks_raster = src.read()
                masks_raster[masks_raster==2] = 0  # fix the cloud pixels
                masks_raster[masks_raster==0] = 255  # revert the mask
                masks_raster[masks_raster!=255] = 0  # change the other pixels to 0
                masks['Red'] = masks['Green'] = masks['Blue'] = masks_raster.copy()
            else:
                i = i.replace('BGRN_SR', 'udm2')
                with rasterio.open(i, 'r') as src:
                    masks_raster = src.read()
                final_mask = masks_raster[0]+masks_raster[2]+masks_raster[5]
                masks['Red'] = masks['Green'] = masks['Blue'] = final_mask[np.newaxis, :, :]
    
    # Plot histograms for patches bands
    patches = extract_patches(raster=np.concatenate([bands['Red'], bands['Green'], bands['Blue']]),
                              mask=np.concatenate([masks['Red'], masks['Green'], masks['Blue']]), dim=dim)
    fig, axs = plt.subplots(4, 4, figsize=(48, 48))
    r = np.random.RandomState(42)
    idxs = r.choice(len(patches), 4)
    colors = {'Red': 'r', 'Green': 'g', 'Blue': 'b'}
    for i, idx in enumerate(idxs):
        axs[0][i].imshow(quantile_scaling(np.moveaxis(patches[idx], 0, 2), 'uniform'))  # let's quantize just for visualise stuff
        axs[0][i].set_title(f'Bit depth {patches[idx].dtype}', fontsize=fsize)
        axs[0][i].axis('off')
        for j, color in enumerate(['Red', 'Green', 'Blue']):
            # Select the band
            band = patches[idx][j]
            # Consider different scales for plotting the PMF
            q_max = np.quantile(band.flatten(), 0.99)  # Let's pick the 99th quantile

            # Plot everything
            axs[1][i].hist(band.flatten(), 
                        bins=200, range=(0, np.iinfo(dtype).max), color=colors[color], 
                        label=color, alpha=0.8, density=True)
            axs[1][i].set_title('Standard distribution', fontsize=fsize)
            axs[1][i].tick_params(axis='both', labelsize=fsize)
            axs[1][i].legend(fontsize=fsize)
            axs[2][i].hist(band.flatten(), 
                        bins=200, range=(0, q_max), color=colors[color], 
                        label=color, alpha=0.8, density=True)
            axs[2][i].set_title('0-99th quantile distribution', fontsize=fsize)
            axs[2][i].tick_params(axis='both', labelsize=fsize)
            axs[2][i].legend(fontsize=fsize)
            axs[3][i].hist(band.flatten(), 
                        bins=200, range=(0, band.max()), color=colors[color], 
                        label=color, alpha=0.8, density=True)
            axs[3][i].set_title('0-max distribution', fontsize=fsize)
            axs[3][i].tick_params(axis='both', labelsize=fsize)
            axs[3][i].legend(fontsize=fsize)
    fig.suptitle(f"{product} bands and patches, {dtype} bits depth", fontsize=fsize+5)
    plt.show()


# ### Let's see the values from the normalization techniques suggested by Paolo

# In[20]:


save_path = '/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/patches'

dim = (3, 256, 256)
fsize = 30
for product in data_df.drop('CBERS_4_PAN10M_20220905_151_116').index.get_level_values(0).unique():
#for product in ['RapidEye4_Sao_Miguel_Arcangeo_reorthotile_analytic_sr', 'RapidEye2_Santos_reorthotile_analytic_sr', 'PlanetScope_Campinas_psorthotile_analytic_sr_udm2']:
    
    # Select the product
    df = data_df.loc[product]
    
    bands = dict()
    masks = dict()
    robust_scalers99 = dict()
    robust_scalers95 = dict()
    for i, r in df.iterrows():
        if r['bands_name'] in ['Red', 'Green', 'Blue']:
            # Load the band and mask
            with rasterio.open(i, 'r') as src:
                band = src.read()
                mask = src.read_masks()
                dtype = src.profile['dtype']
            
            bands[r['bands_name']] = band
            # Add check for CBERS2 and Landsat, their masks are not reliable
            if product not in ['CBERS_2_CCD1XS_20081230_156_116', 'CBERS_2B_CCD1XS_20090905_167_120', 'LO82240772021365CUB00']:
                masks[r['bands_name']] = mask
            else:
                masks[r['bands_name']] = (band!=0).astype(np.uint8)  
                
            # Train the RobustScalers
            robust_scalers99[r['bands_name']] = RobustScaler(quantile_range=(0, 99)).fit(band[mask.astype(np.bool)].reshape(-1, 1))
            band_norm = robust_scalers99[r['bands_name']].transform(band[mask.astype(np.bool)].reshape(-1, 1))
            robust_scalers99[f"{r['bands_name']}_min"], robust_scalers99[f"{r['bands_name']}_max"] = band_norm.min(), band_norm.max()
            robust_scalers95[r['bands_name']] = RobustScaler(quantile_range=(0, 95)).fit(band[mask.astype(np.bool)].reshape(-1, 1))
            band_norm = robust_scalers95[r['bands_name']].transform(band[mask.astype(np.bool)].reshape(-1, 1))
            robust_scalers95[f"{r['bands_name']}_min"], robust_scalers95[f"{r['bands_name']}_max"] = band_norm.min(), band_norm.max()
                
        elif r['bands_name'] in ['RGBN', 'RGBReN']:
            
            # Load bands
            with rasterio.open(i, 'r') as src:
                raster = src.read()
                dtype = src.profile['dtype']
            # Add the bands to the bands dictionary
            bands['Red'] = raster[2][np.newaxis, :, :]
            bands['Green'] = raster[1][np.newaxis, :, :]
            bands['Blue'] = raster[0][np.newaxis, :, :]
            
            # Load masks
            if 'RapidEye' in i:
                i = i.replace('Analytic_SR', 'udm')
                with rasterio.open(i, 'r') as src:
                    masks_raster = src.read()
                masks_raster[masks_raster==2] = 0  # fix the cloud pixels
                masks_raster[masks_raster==0] = 255  # revert the mask
                masks_raster[masks_raster!=255] = 0  # change the other pixels to 0
                masks['Red'] = masks['Green'] = masks['Blue'] = masks_raster.copy()
            else:
                i = i.replace('BGRN_SR', 'udm2')
                with rasterio.open(i, 'r') as src:
                    masks_raster = src.read()
                final_mask = masks_raster[0]+masks_raster[2]+masks_raster[5]
                masks['Red'] = masks['Green'] = masks['Blue'] = final_mask[np.newaxis, :, :]
                
            # Train the RobustScalers
            for color in ['Red', 'Green', 'Blue']:
                robust_scalers99[color] = RobustScaler(quantile_range=(0, 99)).fit(bands[color][masks[color].astype(np.bool)].reshape(-1, 1))
                band_norm = robust_scalers99[color].transform(bands[color][masks[color].astype(np.bool)].reshape(-1, 1))
                robust_scalers99[f"{color}_min"], robust_scalers99[f"{color}_max"] = band_norm.min(), band_norm.max()
                robust_scalers95[color] = RobustScaler(quantile_range=(0, 95)).fit(bands[color][masks[color].astype(np.bool)].reshape(-1, 1))
                band_norm = robust_scalers95[color].transform(bands[color][masks[color].astype(np.bool)].reshape(-1, 1))
                robust_scalers95[f"{color}_min"], robust_scalers95[f"{color}_max"] = band_norm.min(), band_norm.max()
                
    
    # Train the mean Robustscalers
    robust_scalers99['Mean'] = RobustScaler(quantile_range=(0, 99)).fit(np.mean(np.concatenate([bands['Red'], bands['Green'], bands['Blue']]), axis=0).reshape(-1, 1))
    band_norm = robust_scalers99['Mean'].transform(np.mean(np.concatenate([bands['Red'], bands['Green'], bands['Blue']]), axis=0).reshape(-1, 1))
    robust_scalers99['Mean_min'], robust_scalers99['Mean_max'] = band_norm.min(), band_norm.max()
    robust_scalers95['Mean'] = RobustScaler(quantile_range=(0, 95)).fit(np.mean(np.concatenate([bands['Red'], bands['Green'], bands['Blue']]), axis=0).reshape(-1, 1))
    band_norm = robust_scalers95['Mean'].transform(np.mean(np.concatenate([bands['Red'], bands['Green'], bands['Blue']]), axis=0).reshape(-1, 1))
    robust_scalers95['Mean_min'], robust_scalers95['Mean_max'] = band_norm.min(), band_norm.max()
    
    # Plot histograms for patches bands
    patches = extract_patches(raster=np.concatenate([bands['Red'], bands['Green'], bands['Blue']]),
                              mask=np.concatenate([masks['Red'], masks['Green'], masks['Blue']]), dim=dim)
    fig, axs = plt.subplots(4, 6, figsize=(48, 48))
    r = np.random.RandomState(42)
    idxs = r.choice(len(patches), 4)
    colors = {'Red': 'r', 'Green': 'g', 'Blue': 'b'}
    for i, idx in enumerate(idxs):
        axs[i][0].imshow(quantile_scaling(np.moveaxis(patches[idx], 0, 2), 'uniform'))  # let's quantize just for visualise stuff
        axs[i][0].set_title(f'Bit depth {patches[idx].dtype}', fontsize=fsize)
        axs[i][0].axis('off')
        for j, color in enumerate(['Red', 'Green', 'Blue']):
            # Select the band
            band = patches[idx][j]
            # Consider different scales for plotting the PMF
            band_99 = robust_scalers99[color].transform(band.reshape(-1, 1)).reshape(band.shape)
            band_99 = (band_99-robust_scalers99[f"{color}_min"])/(robust_scalers99[f"{color}_max"]-robust_scalers99[f"{color}_min"])
            band_99_mean = robust_scalers99['Mean'].transform(band.reshape(-1, 1)).reshape(band.shape)
            band_99_mean = (band_99-robust_scalers99[f"Mean_min"])/(robust_scalers99[f"Mean_max"]-robust_scalers99[f"Mean_min"])
            band_95 = robust_scalers95[color].transform(band.reshape(-1, 1)).reshape(band.shape)
            band_95 = (band_95-robust_scalers95[f"{color}_min"])/(robust_scalers95[f"{color}_max"]-robust_scalers95[f"{color}_min"]) 
            band_95_mean = robust_scalers95['Mean'].transform(band.reshape(-1, 1)).reshape(band.shape)
            band_95_mean = (band_95-robust_scalers95[f"Mean_min"])/(robust_scalers95[f"Mean_max"]-robust_scalers95[f"Mean_min"])
            
            # Plot everything
            axs[i][1].hist(band.flatten(), 
                        bins=200, range=(0, np.iinfo(dtype).max), color=colors[color], 
                        label=color, alpha=0.8, density=True)
            axs[i][1].set_title('Standard distribution', fontsize=fsize)
            axs[i][1].tick_params(axis='both', labelsize=fsize)
            axs[i][1].legend(fontsize=fsize)
            axs[i][2].hist(band_99.flatten(), 
                        bins=200, color=colors[color], 
                        label=color, alpha=0.8, density=True)
            axs[i][2].set_title('MinQ99max quantile distribution', fontsize=fsize)
            axs[i][2].tick_params(axis='both', labelsize=fsize)
            axs[i][2].legend(fontsize=fsize)
            axs[i][3].hist(band_99.flatten(), 
                        bins=200, color=colors[color], 
                        label=color, alpha=0.8, density=True)
            axs[i][3].set_title('MeanQ99max quantile distribution', fontsize=fsize)
            axs[i][3].tick_params(axis='both', labelsize=fsize)
            axs[i][3].legend(fontsize=fsize)
            axs[i][4].hist(band_95.flatten(), 
                        bins=200, color=colors[color], 
                        label=color, alpha=0.8, density=True)
            axs[i][4].set_title('MinQ95max quantile distribution', fontsize=fsize)
            axs[i][4].tick_params(axis='both', labelsize=fsize)
            axs[i][4].legend(fontsize=fsize)
            axs[i][5].hist(band_95_mean.flatten(), 
                        bins=200, color=colors[color], 
                        label=color, alpha=0.8, density=True)
            axs[i][5].set_title('MeanQ95max quantile distribution', fontsize=fsize)
            axs[i][5].tick_params(axis='both', labelsize=fsize)
            axs[i][5].legend(fontsize=fsize)
    fig.suptitle(f"{product} bands and patches, {dtype} bits depth", fontsize=fsize+5)
    plt.show()


# ### Let's plot one example of mean distribution

# In[19]:


fig, axs = plt.subplots(2, 1, figsize=(24, 18))
axs[0].imshow(quantile_scaling(np.moveaxis(patches[idx], 0, 2), 'uniform'))  # let's quantize just for visualise stuff
axs[0].set_title(f'Bit depth {patches[idx].dtype}', fontsize=fsize)
axs[0].axis('off')
colors = {'Red': 'r', 'Green': 'g', 'Blue': 'b'}
for j, color in enumerate(['Red', 'Green', 'Blue']):
    axs[1].hist(patches[idx][j].flatten(), bins=200, range=(0, band.max()), color=colors[color], 
                label=color, alpha=0.8, density=True)
axs[1].hist(np.mean(patches[idx], axis=0).flatten(), bins=200, range=(0, patches[idx].max()), color='orange', alpha=0.8,
           label='Mean intensity', density=True)
axs[1].tick_params(axis='both', labelsize=fsize)
axs[1].legend(fontsize=fsize)


# In[18]:


np.mean(patches[idx], axis=0).shape


# ### Let's save them

# In[8]:


get_ipython().run_cell_magic('javascript', '', '(function(on) {\nconst e=$( "<a>Setup failed</a>" );\nconst ns="js_jupyter_suppress_warnings";\nvar cssrules=$("#"+ns);\nif(!cssrules.length) cssrules = $("<style id=\'"+ns+"\' type=\'text/css\'>div.output_stderr { } </style>").appendTo("head");\ne.click(function() {\n    var s=\'Showing\';  \n    cssrules.empty()\n    if(on) {\n        s=\'Hiding\';\n        cssrules.append("div.output_stderr, div[data-mime-type*=\'.stderr\'] { display:none; }");\n    }\n    e.text(s+\' warnings (click to toggle)\');\n    on=!on;\n}).click();\n$(element).append(e);\n})(true);\n')


# In[29]:


save_path = '/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/patches'

dim = (3, 256, 256)
fsize = 30
for product in data_df.drop('CBERS_4_PAN10M_20220905_151_116').index.get_level_values(0).unique():
#for product in ['RapidEye4_Sao_Miguel_Arcangeo_reorthotile_analytic_sr', 'RapidEye2_Santos_reorthotile_analytic_sr', 'PlanetScope_Campinas_psorthotile_analytic_sr_udm2']:
    
    # Select the product
    df = data_df.loc[product]
    
    # Load the bands
    bands = dict()
    masks = dict()
    for i, r in df.iterrows():
        if r['bands_name'] in ['Red', 'Green', 'Blue']:
            # Load the band and mask
            with rasterio.open(i, 'r') as src:
                band = src.read()
                mask = src.read_masks()
                dtype = src.profile['dtype']
                profile = src.profile
            
            bands[r['bands_name']] = band
            # Add check for CBERS2 and Landsat, their masks are not reliable
            if product not in ['CBERS_2_CCD1XS_20081230_156_116', 'CBERS_2B_CCD1XS_20090905_167_120', 'LO82240772021365CUB00']:
                masks[r['bands_name']] = mask
            else:
                masks[r['bands_name']] = (band!=0).astype(np.uint8)  
                
        elif r['bands_name'] in ['RGBN', 'RGBReN']:
            
            # Load bands
            with rasterio.open(i, 'r') as src:
                raster = src.read()
                dtype = src.profile['dtype']
                profile = src.profile
            # Add the bands to the bands dictionary
            bands['Red'] = raster[2][np.newaxis, :, :]
            bands['Green'] = raster[1][np.newaxis, :, :]
            bands['Blue'] = raster[0][np.newaxis, :, :]
            
            # Load masks
            if 'RapidEye' in i:
                i = i.replace('Analytic_SR', 'udm')
                with rasterio.open(i, 'r') as src:
                    masks_raster = src.read()
                masks_raster[masks_raster==2] = 0  # fix the cloud pixels
                masks_raster[masks_raster==0] = 255  # revert the mask
                masks_raster[masks_raster!=255] = 0  # change the other pixels to 0
                masks['Red'] = masks['Green'] = masks['Blue'] = masks_raster.copy()
            else:
                i = i.replace('BGRN_SR', 'udm2')
                with rasterio.open(i, 'r') as src:
                    masks_raster = src.read()
                final_mask = masks_raster[0]+masks_raster[2]+masks_raster[5]
                masks['Red'] = masks['Green'] = masks['Blue'] = final_mask[np.newaxis, :, :]
    
    # Extract patches
    patches = extract_patches(raster=np.concatenate([bands['Red'], bands['Green'], bands['Blue']]),
                              mask=np.concatenate([masks['Red'], masks['Green'], masks['Blue']]), dim=dim)
    # Select the same indexes for plotting
    r = np.random.RandomState(42)
    idxs = r.choice(len(patches), 4)
    # Save them
    product_save_path = os.path.join(save_path, product)
    os.makedirs(product_save_path, exist_ok=True)
    for idx, patch in enumerate(patches):
        # Change the profile info
        profile['height'] = 256
        profile['width'] = 256
        profile['count'] = 3
        # Save the raster data
        with rasterio.open(os.path.join(product_save_path, f'{idx}_patch.tiff'), 'w', **profile) as dst:
            dst.write(patch)
            
            
    # Let's reload the patches and plot them to see if the histogram matches with the previous ones
    patches_paths = sorted(glob.glob(os.path.join(product_save_path, '*_patch.tiff')))
    patches = {int(os.path.split(path)[-1].split('_')[0]): None for path in patches_paths}
    for path in patches_paths:
        with rasterio.open(path, 'r') as src:
            patches[int(os.path.split(path)[-1].split('_')[0])] = src.read()
    # Plot histograms for patches bands
    fig, axs = plt.subplots(4, 4, figsize=(48, 48))
    colors = {'Red': 'r', 'Green': 'g', 'Blue': 'b'}
    for i, idx in enumerate(idxs):
        axs[0][i].imshow(quantile_scaling(np.moveaxis(patches[idx], 0, 2), 'uniform'))  # let's quantize just for visualise stuff
        axs[0][i].set_title(f'Bit depth {patches[idx].dtype}', fontsize=fsize)
        axs[0][i].axis('off')
        for j, color in enumerate(['Red', 'Green', 'Blue']):
            # Select the band
            band = patches[idx][j]
            # Consider different scales for plotting the PMF
            q_max = np.quantile(band.flatten(), 0.99)  # Let's pick the 99th quantile

            # Plot everything
            axs[1][i].hist(band.flatten(), 
                        bins=200, range=(0, np.iinfo(dtype).max), color=colors[color], 
                        label=color, alpha=0.8, density=True)
            axs[1][i].set_title('Standard distribution', fontsize=fsize)
            axs[1][i].tick_params(axis='both', labelsize=fsize)
            axs[1][i].legend(fontsize=fsize)
            axs[2][i].hist(band.flatten(), 
                        bins=200, range=(0, q_max), color=colors[color], 
                        label=color, alpha=0.8, density=True)
            axs[2][i].set_title('0-99th quantile distribution', fontsize=fsize)
            axs[2][i].tick_params(axis='both', labelsize=fsize)
            axs[2][i].legend(fontsize=fsize)
            axs[3][i].hist(band.flatten(), 
                        bins=200, range=(0, band.max()), color=colors[color], 
                        label=color, alpha=0.8, density=True)
            axs[3][i].set_title('0-max distribution', fontsize=fsize)
            axs[3][i].tick_params(axis='both', labelsize=fsize)
            axs[3][i].legend(fontsize=fsize)
    fig.suptitle(f"{product} bands and patches, {dtype} bits depth", fontsize=fsize+5)
    plt.show()


# ### Let's save the RobustScalers for the original products, and then apply them to the patches
# We want to see the same distribution of values.

# In[53]:


from sklearn.preprocessing import RobustScaler
save_path = '/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/patches'

dim = (3, 256, 256)
fsize = 30
for product in data_df.drop('CBERS_4_PAN10M_20220905_151_116').index.get_level_values(0).unique():
#for product in ['RapidEye4_Sao_Miguel_Arcangeo_reorthotile_analytic_sr', 'RapidEye2_Santos_reorthotile_analytic_sr', 'PlanetScope_Campinas_psorthotile_analytic_sr_udm2']:
    
    # Select the product
    df = data_df.loc[product]
    
    # Load the bands
    bands = dict()
    masks = dict()
    robust_scalers99 = dict()
    robust_scalers95 = dict()
    for i, r in df.iterrows():
        if r['bands_name'] in ['Red', 'Green', 'Blue']:
            # Load the band and mask
            with rasterio.open(i, 'r') as src:
                band = src.read()
                mask = src.read_masks()
                dtype = src.profile['dtype']
                profile = src.profile
            
            bands[r['bands_name']] = band
            # Add check for CBERS2 and Landsat, their masks are not reliable
            if product not in ['CBERS_2_CCD1XS_20081230_156_116', 'CBERS_2B_CCD1XS_20090905_167_120', 'LO82240772021365CUB00']:
                masks[r['bands_name']] = mask
            else:
                masks[r['bands_name']] = (band!=0).astype(np.uint8)
                
            # Train the RobustScalers
            robust_scalers99[r['bands_name']] = RobustScaler(quantile_range=(0, 99)).fit(band[mask.astype(np.bool)].reshape(-1, 1))
            band_norm = robust_scalers99[r['bands_name']].transform(band[mask.astype(np.bool)].reshape(-1, 1))
            robust_scalers99[f"{r['bands_name']}_min"], robust_scalers99[f"{r['bands_name']}_max"] = band_norm.min(), band_norm.max()
            robust_scalers95[r['bands_name']] = RobustScaler(quantile_range=(0, 95)).fit(band[mask.astype(np.bool)].reshape(-1, 1))
            band_norm = robust_scalers95[r['bands_name']].transform(band[mask.astype(np.bool)].reshape(-1, 1))
            robust_scalers95[f"{r['bands_name']}_min"], robust_scalers95[f"{r['bands_name']}_max"] = band_norm.min(), band_norm.max()
                
        elif r['bands_name'] in ['RGBN', 'RGBReN']:
            
            # Load bands
            with rasterio.open(i, 'r') as src:
                raster = src.read()
                dtype = src.profile['dtype']
                profile = src.profile
            # Add the bands to the bands dictionary
            bands['Red'] = raster[2][np.newaxis, :, :]
            bands['Green'] = raster[1][np.newaxis, :, :]
            bands['Blue'] = raster[0][np.newaxis, :, :]
            
            # Load masks
            if 'RapidEye' in i:
                i = i.replace('Analytic_SR', 'udm')
                with rasterio.open(i, 'r') as src:
                    masks_raster = src.read()
                masks_raster[masks_raster==2] = 0  # fix the cloud pixels
                masks_raster[masks_raster==0] = 255  # revert the mask
                masks_raster[masks_raster!=255] = 0  # change the other pixels to 0
                masks['Red'] = masks['Green'] = masks['Blue'] = masks_raster.copy()
            else:
                i = i.replace('BGRN_SR', 'udm2')
                with rasterio.open(i, 'r') as src:
                    masks_raster = src.read()
                final_mask = masks_raster[0]+masks_raster[2]+masks_raster[5]
                masks['Red'] = masks['Green'] = masks['Blue'] = final_mask[np.newaxis, :, :]
                
            # Train the RobustScalers
            for color in ['Red', 'Green', 'Blue']:
                robust_scalers99[color] = RobustScaler(quantile_range=(0, 99)).fit(bands[color][masks[color].astype(np.bool)].reshape(-1, 1))
                band_norm = robust_scalers99[color].transform(bands[color][masks[color].astype(np.bool)].reshape(-1, 1))
                robust_scalers99[f"{color}_min"], robust_scalers99[f"{color}_max"] = band_norm.min(), band_norm.max()
                robust_scalers95[color] = RobustScaler(quantile_range=(0, 95)).fit(bands[color][masks[color].astype(np.bool)].reshape(-1, 1))
                band_norm = robust_scalers95[color].transform(bands[color][masks[color].astype(np.bool)].reshape(-1, 1))
                robust_scalers95[f"{color}_min"], robust_scalers95[f"{color}_max"] = band_norm.min(), band_norm.max()
    
    # Train the mean band quantile scaler
    robust_scalers99['Mean'] = RobustScaler(quantile_range=(0, 99)).fit(np.mean(np.concatenate([bands['Red'], bands['Green'], bands['Blue']]), axis=0).reshape(-1, 1))
    robust_scalers99['Mean_min'], robust_scalers99['Mean_max'] = robust_scalers99['Mean'].transform(np.mean(np.concatenate([bands['Red'], bands['Green'], bands['Blue']]), axis=0).reshape(-1, 1)).min(), robust_scalers99['Mean'].transform(np.mean(np.concatenate([bands['Red'], bands['Green'], bands['Blue']]), axis=0).reshape(-1, 1)).max()
    robust_scalers95['Mean'] = RobustScaler(quantile_range=(0, 95)).fit(np.mean(np.concatenate([bands['Red'], bands['Green'], bands['Blue']]), axis=0).reshape(-1, 1))
    robust_scalers95['Mean_min'], robust_scalers95['Mean_max'] = robust_scalers95['Mean'].transform(np.mean(np.concatenate([bands['Red'], bands['Green'], bands['Blue']]), axis=0).reshape(-1, 1)).min(), robust_scalers95['Mean'].transform(np.mean(np.concatenate([bands['Red'], bands['Green'], bands['Blue']]), axis=0).reshape(-1, 1)).max()
    # Plot histograms for patches bands
    patches = extract_patches(raster=np.concatenate([bands['Red'], bands['Green'], bands['Blue']]),
                              mask=np.concatenate([masks['Red'], masks['Green'], masks['Blue']]), dim=dim)
    patches = np.moveaxis(patches, 0, 1)
    # Plot histograms for patches bands
    fig, axs = plt.subplots(2, 4, figsize=(48, 24))
    colors = {'Red': 'r', 'Green': 'g', 'Blue': 'b'}
    colors_idx = {'Green': 1, 'Red': 0, 'Blue': 2}
    for i, color in enumerate(['Green', 'Red', 'Blue']):
        norm_patches = robust_scalers99[color].transform(patches[colors_idx[color]].reshape(-1, 1)).reshape(patches[colors_idx[color]].shape)
        norm_patches = (norm_patches-robust_scalers99[f'{color}_min'])/(robust_scalers99[f'{color}_max']-robust_scalers99[f'{color}_min'])
        axs[0][0].hist(norm_patches.flatten(), 
                       bins=200, range=(0, 1), color=colors[color], 
                       label=color, alpha=0.8, density=True)
        axs[0][0].set_title('MinQMax (0-99th quantile) patches scaling distribution', fontsize=fsize)
        axs[0][0].tick_params(axis='both', labelsize=fsize)
        axs[0][0].legend(fontsize=fsize)
        norm_patches = robust_scalers99['Mean'].transform(patches[colors_idx[color]].reshape(-1, 1)).reshape(patches[colors_idx[color]].shape)
        norm_patches = (norm_patches-robust_scalers99[f'Mean_min'])/(robust_scalers99[f'Mean_max']-robust_scalers99[f'Mean_min'])
        axs[0][1].hist(norm_patches.flatten(), 
                    bins=200, range=(0, 1), color=colors[color], 
                    label=color, alpha=0.8, density=True)
        axs[0][1].set_title('MeanMinQMax (0-99th quantile) patches scaling distribution', fontsize=fsize)
        axs[0][1].tick_params(axis='both', labelsize=fsize)
        axs[0][1].legend(fontsize=fsize)
        norm_patches = robust_scalers95[color].transform(patches[colors_idx[color]].reshape(-1, 1)).reshape(patches[colors_idx[color]].shape)
        norm_patches = (norm_patches-robust_scalers95[f'{color}_min'])/(robust_scalers95[f'{color}_max']-robust_scalers95[f'{color}_min'])
        axs[0][2].hist(norm_patches.flatten(), 
                    bins=200, range=(0, 1), color=colors[color], 
                    label=color, alpha=0.8, density=True)
        axs[0][2].set_title('MinQMax (0-95th quantile) patches scaling distribution', fontsize=fsize)
        axs[0][2].tick_params(axis='both', labelsize=fsize)
        axs[0][2].legend(fontsize=fsize)
        norm_patches = robust_scalers95['Mean'].transform(patches[colors_idx[color]].reshape(-1, 1)).reshape(patches[colors_idx[color]].shape)
        norm_patches = (norm_patches-robust_scalers95[f'Mean_min'])/(robust_scalers99[f'Mean_max']-robust_scalers99[f'Mean_min'])
        axs[0][3].hist(norm_patches.flatten(), 
                    bins=200, range=(0, 1), color=colors[color], 
                    label=color, alpha=0.8, density=True)
        axs[0][3].set_title('MeanMinQMax (0-95th quantile) patches scaling distribution', fontsize=fsize)
        axs[0][3].tick_params(axis='both', labelsize=fsize)
        axs[0][3].legend(fontsize=fsize)
    
    # Let's reload the patches and plot them to see if the histogram matches with the previous ones
    product_save_path = os.path.join(save_path, product)
    patches_paths = sorted(glob.glob(os.path.join(product_save_path, '*_patch.tiff')))
    patches = []
    for path in patches_paths:
        with rasterio.open(path, 'r') as src:
            patches.append(src.read())
    # Plot histograms for patches bands
    patches = np.moveaxis(np.array(patches), 0, 1)
    colors_idx = {'Green': 1, 'Red': 0, 'Blue': 2}
    for i, color in enumerate(['Green', 'Red', 'Blue']):
        norm_patches = robust_scalers99[color].transform(patches[colors_idx[color]].reshape(-1, 1)).reshape(patches[colors_idx[color]].shape)
        norm_patches = (norm_patches-robust_scalers99[f'{color}_min'])/(robust_scalers99[f'{color}_max']-robust_scalers99[f'{color}_min'])
        axs[1][0].hist(norm_patches.flatten(), 
                       bins=200, range=(0, 1), color=colors[color], 
                       label=color, alpha=0.8, density=True)
        axs[1][0].set_title('MinQMax (0-99th quantile) patches scaling distribution', fontsize=fsize)
        axs[1][0].tick_params(axis='both', labelsize=fsize)
        axs[1][0].legend(fontsize=fsize)
        norm_patches = robust_scalers99['Mean'].transform(patches[colors_idx[color]].reshape(-1, 1)).reshape(patches[colors_idx[color]].shape)
        norm_patches = (norm_patches-robust_scalers99[f'Mean_min'])/(robust_scalers99[f'Mean_max']-robust_scalers99[f'Mean_min'])
        axs[1][1].hist(norm_patches.flatten(), 
                    bins=200, range=(0, 1), color=colors[color], 
                    label=color, alpha=0.8, density=True)
        axs[1][1].set_title('MeanMinQMax (0-99th quantile) patches scaling distribution', fontsize=fsize)
        axs[1][1].tick_params(axis='both', labelsize=fsize)
        axs[1][1].legend(fontsize=fsize)
        norm_patches = robust_scalers95[color].transform(patches[colors_idx[color]].reshape(-1, 1)).reshape(patches[colors_idx[color]].shape)
        norm_patches = (norm_patches-robust_scalers95[f'{color}_min'])/(robust_scalers95[f'{color}_max']-robust_scalers95[f'{color}_min'])
        axs[1][2].hist(norm_patches.flatten(), 
                    bins=200, range=(0, 1), color=colors[color], 
                    label=color, alpha=0.8, density=True)
        axs[1][2].set_title('MinQMax (0-95th quantile) patches scaling distribution', fontsize=fsize)
        axs[1][2].tick_params(axis='both', labelsize=fsize)
        axs[1][2].legend(fontsize=fsize)
        norm_patches = robust_scalers95['Mean'].transform(patches[colors_idx[color]].reshape(-1, 1)).reshape(patches[colors_idx[color]].shape)
        norm_patches = (norm_patches-robust_scalers95[f'Mean_min'])/(robust_scalers99[f'Mean_max']-robust_scalers99[f'Mean_min'])
        axs[1][3].hist(norm_patches.flatten(), 
                    bins=200, range=(0, 1), color=colors[color], 
                    label=color, alpha=0.8, density=True)
        axs[1][3].set_title('MeanMinQMax (0-95th quantile) patches scaling distribution', fontsize=fsize)
        axs[1][3].tick_params(axis='both', labelsize=fsize)
        axs[1][3].legend(fontsize=fsize)
    fig.suptitle(f"{product} bands and PMF, {dtype} bits depth", fontsize=fsize)
    plt.show()


# ### OK, everything is working! Let's just save the RobustScalers and MaxScalers and statistics for normalzing the patches during training

# ### Let's define our custom RobustScaler in order to save the properties we need for each satellite product

# In[11]:


from sklearn.preprocessing import RobustScaler
from abc import ABC, abstractmethod

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
        if img.shape[0] > 3:
            raise RuntimeError('Normalization works only for RGB images, sorry!')
        else:
            img = img.astype(np.float32)
            for i, band in enumerate(['Red', 'Green', 'Blue']):
                if mean_scaling:
                    img[i] /= self.max['Mean']
                else:
                    img[i] /= self.max[band]
        # Move the channel as the last dimension
        min_dim = np.argmin(img.shape)
        img = np.moveaxis(img, min_dim, -1)
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

    def __init__(self, *, product: str = None):
        self.product = product
        self.scalers = dict()
        self.min = dict()
        self.max = dict()

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
        if img.shape[0] > 3:
            raise RuntimeError('Normalization works only for RGB images, sorry!')
        else:
            img = img.astype(np.float32)
            if mean_scaling:
                for i in range(img.shape[0]):
                    img[i] = self.scalers['Mean'].transform(img[i].reshape(-1, 1)).reshape(img[i].shape)
                    img[i] = ((img[i] - self.min['Mean']) / (self.max['Mean'] - self.min['Mean']))
            else:
                for i, band in enumerate(['Red', 'Green', 'Blue']):
                    img[i] = self.scalers[band].transform(img[0].reshape(-1, 1)).reshape(img[0].shape)
                    img[i] = ((img[i] - self.min[band]) / (self.max[band] - self.min[band]))
        # Move the channel as the last dimension
        min_dim = np.argmin(img.shape)
        img = np.moveaxis(img, min_dim, -1)
        return img

    def open_and_normalize_product(self, path: str, method: bool = True) -> np.array:
        # Open the product
        with rasterio.open(path, 'r') as src:
            img = src.read()
        # Normalize it
        img = self.normalize_product(img, method)
        return img


# ### RobustScaler

# In[8]:


from joblib import dump
from tqdm.notebook import tqdm
save_path = '/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/patches'

dim = (3, 256, 256)
fsize = 30
for product in tqdm(data_df.drop('CBERS_4_PAN10M_20220905_151_116').index.get_level_values(0).unique()):
#for product in ['RapidEye4_Sao_Miguel_Arcangeo_reorthotile_analytic_sr', 'RapidEye2_Santos_reorthotile_analytic_sr', 'PlanetScope_Campinas_psorthotile_analytic_sr_udm2']:
    
    # Select the product
    df = data_df.loc[product]
    
    # Instantiate the Scaler and the product saving path
    product_save_path = os.path.join(save_path, product)
    scaler99 = SatProductRobustScaler(product=product)
    scaler95 = SatProductRobustScaler(product=product)
    
    # Load the bands
    bands = dict()
    masks = dict()
    for i, r in df.iterrows():
        if r['bands_name'] in ['Red', 'Green', 'Blue']:
            # Load the band and mask
            with rasterio.open(i, 'r') as src:
                band = src.read()
                mask = src.read_masks()
                dtype = src.profile['dtype']
                profile = src.profile
            
            bands[r['bands_name']] = band
            # Add check for CBERS2 and Landsat, their masks are not reliable
            if product not in ['CBERS_2_CCD1XS_20081230_156_116', 'CBERS_2B_CCD1XS_20090905_167_120', 'LO82240772021365CUB00']:
                masks[r['bands_name']] = mask
            else:
                masks[r['bands_name']] = (band!=0).astype(np.uint8)
                
            # Train the RobustScalers and save them
            scaler = RobustScaler(quantile_range=(0, 99)).fit(band[mask.astype(np.bool)].reshape(-1, 1))
            band_norm = scaler.transform(band[mask.astype(np.bool)].reshape(-1, 1))
            scaler99.scalers[r['bands_name']] = scaler
            scaler99.min[r['bands_name']], scaler99.max[r['bands_name']] = band_norm.min(), band_norm.max()
            scaler = RobustScaler(quantile_range=(0, 95)).fit(band[mask.astype(np.bool)].reshape(-1, 1))
            band_norm = scaler.transform(band[mask.astype(np.bool)].reshape(-1, 1))
            scaler95.scalers[r['bands_name']] = scaler
            scaler95.min[r['bands_name']], scaler95.max[r['bands_name']] = band_norm.min(), band_norm.max()
                
        elif r['bands_name'] in ['RGBN', 'RGBReN']:
            
            # Load bands
            with rasterio.open(i, 'r') as src:
                raster = src.read()
                dtype = src.profile['dtype']
                profile = src.profile
            # Add the bands to the bands dictionary
            bands['Red'] = raster[2][np.newaxis, :, :]
            bands['Green'] = raster[1][np.newaxis, :, :]
            bands['Blue'] = raster[0][np.newaxis, :, :]
            
            # Load masks
            if 'RapidEye' in i:
                i = i.replace('Analytic_SR', 'udm')
                with rasterio.open(i, 'r') as src:
                    masks_raster = src.read()
                masks_raster[masks_raster==2] = 0  # fix the cloud pixels
                masks_raster[masks_raster==0] = 255  # revert the mask
                masks_raster[masks_raster!=255] = 0  # change the other pixels to 0
                masks['Red'] = masks['Green'] = masks['Blue'] = masks_raster.copy()
            else:
                i = i.replace('BGRN_SR', 'udm2')
                with rasterio.open(i, 'r') as src:
                    masks_raster = src.read()
                final_mask = masks_raster[0]+masks_raster[2]+masks_raster[5]
                masks['Red'] = masks['Green'] = masks['Blue'] = final_mask[np.newaxis, :, :]
                
            # Train the RobustScalers
            for color in ['Red', 'Green', 'Blue']:
                scaler = RobustScaler(quantile_range=(0, 99)).fit(bands[color][masks[color].astype(np.bool)].reshape(-1, 1))
                band_norm = scaler.transform(bands[color][masks[color].astype(np.bool)].reshape(-1, 1))
                scaler99.scalers[color] = scaler
                scaler99.min[color], scaler99.max[color] = band_norm.min(), band_norm.max()
                scaler = RobustScaler(quantile_range=(0, 95)).fit(bands[color][masks[color].astype(np.bool)].reshape(-1, 1))
                band_norm = scaler.transform(bands[color][masks[color].astype(np.bool)].reshape(-1, 1))
                scaler95.scalers[color] = scaler
                scaler95.min[color], scaler95.max[color] = band_norm.min(), band_norm.max()
    
    # Train the mean band quantile scaler
    scaler99.scalers['Mean'] = RobustScaler(quantile_range=(0, 99)).fit(np.mean(np.concatenate([bands['Red'], bands['Green'], bands['Blue']]), axis=0).reshape(-1, 1))
    scaler99.min['Mean'], scaler99.max['Mean'] = scaler99.scalers['Mean'].transform(np.mean(np.concatenate([bands['Red'], bands['Green'], bands['Blue']]), axis=0).reshape(-1, 1)).min(), scaler99.scalers['Mean'].transform(np.mean(np.concatenate([bands['Red'], bands['Green'], bands['Blue']]), axis=0).reshape(-1, 1)).max()
    scaler95.scalers['Mean'] = RobustScaler(quantile_range=(0, 95)).fit(np.mean(np.concatenate([bands['Red'], bands['Green'], bands['Blue']]), axis=0).reshape(-1, 1))
    scaler95.min['Mean'], scaler95.max['Mean'] = scaler95.scalers['Mean'].transform(np.mean(np.concatenate([bands['Red'], bands['Green'], bands['Blue']]), axis=0).reshape(-1, 1)).min(), scaler95.scalers['Mean'].transform(np.mean(np.concatenate([bands['Red'], bands['Green'], bands['Blue']]), axis=0).reshape(-1, 1)).max()
    dump(scaler99, os.path.join(product_save_path, '99_SatProductRobustScaler.joblib'))
    dump(scaler95, os.path.join(product_save_path, '95_SatProductRobustScaler.joblib'))
    


# ### MaxScaler

# In[12]:


from joblib import dump
from tqdm.notebook import tqdm
save_path = '/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/patches'

dim = (3, 256, 256)
fsize = 30
for product in tqdm(data_df.drop('CBERS_4_PAN10M_20220905_151_116').index.get_level_values(0).unique()):
#for product in ['RapidEye4_Sao_Miguel_Arcangeo_reorthotile_analytic_sr', 'RapidEye2_Santos_reorthotile_analytic_sr', 'PlanetScope_Campinas_psorthotile_analytic_sr_udm2']:
    
    # Select the product
    df = data_df.loc[product]
    
    # Instantiate the Scaler and the product saving path
    product_save_path = os.path.join(save_path, product)
    scaler = SatProductMaxScaler(product=product)
    
    # Load the bands
    bands = dict()
    masks = dict()
    for i, r in df.iterrows():
        if r['bands_name'] in ['Red', 'Green', 'Blue']:
            # Load the band and mask
            with rasterio.open(i, 'r') as src:
                band = src.read()
                mask = src.read_masks()
                dtype = src.profile['dtype']
                profile = src.profile
            
            bands[r['bands_name']] = band
            # Add check for CBERS2 and Landsat, their masks are not reliable
            if product not in ['CBERS_2_CCD1XS_20081230_156_116', 'CBERS_2B_CCD1XS_20090905_167_120', 'LO82240772021365CUB00']:
                masks[r['bands_name']] = mask
            else:
                masks[r['bands_name']] = (band!=0).astype(np.uint8)
                
            # Save the statistics of interest of the scaler
            scaler.min[r['bands_name']], scaler.max[r['bands_name']] = band[mask.astype(np.bool)].min(), band[mask.astype(np.bool)].max()
                
        elif r['bands_name'] in ['RGBN', 'RGBReN']:
            
            # Load bands
            with rasterio.open(i, 'r') as src:
                raster = src.read()
                dtype = src.profile['dtype']
                profile = src.profile
            # Add the bands to the bands dictionary
            bands['Red'] = raster[2][np.newaxis, :, :]
            bands['Green'] = raster[1][np.newaxis, :, :]
            bands['Blue'] = raster[0][np.newaxis, :, :]
            
            # Load masks
            if 'RapidEye' in i:
                i = i.replace('Analytic_SR', 'udm')
                with rasterio.open(i, 'r') as src:
                    masks_raster = src.read()
                masks_raster[masks_raster==2] = 0  # fix the cloud pixels
                masks_raster[masks_raster==0] = 255  # revert the mask
                masks_raster[masks_raster!=255] = 0  # change the other pixels to 0
                masks['Red'] = masks['Green'] = masks['Blue'] = masks_raster.copy()
            else:
                i = i.replace('BGRN_SR', 'udm2')
                with rasterio.open(i, 'r') as src:
                    masks_raster = src.read()
                final_mask = masks_raster[0]+masks_raster[2]+masks_raster[5]
                masks['Red'] = masks['Green'] = masks['Blue'] = final_mask[np.newaxis, :, :]
                
            # Train the RobustScalers
            for color in ['Red', 'Green', 'Blue']:
                scaler.min[color], scaler.max[color] = bands[color][masks[color].astype(np.bool)].min(), bands[color][masks[color].astype(np.bool)].max()
    
    # Train the mean band quantile scaler
    scaler.min['Mean'], scaler.max['Mean'] = np.mean(np.concatenate([bands['Red'], bands['Green'], bands['Blue']]), axis=0).min(), np.mean(np.concatenate([bands['Red'], bands['Green'], bands['Blue']]), axis=0).max()
    dump(scaler, os.path.join(product_save_path, 'SatProductMaxScaler.joblib'))
    


# ### Let's test the scalers to see if everything is going as expected

# In[41]:


from sklearn.preprocessing import RobustScaler
from joblib import load
save_path = '/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/patches'

dim = (3, 256, 256)
fsize = 30
for product in data_df.drop('CBERS_4_PAN10M_20220905_151_116').index.get_level_values(0).unique():
#for product in ['RapidEye4_Sao_Miguel_Arcangeo_reorthotile_analytic_sr', 'RapidEye2_Santos_reorthotile_analytic_sr', 'PlanetScope_Campinas_psorthotile_analytic_sr_udm2']:
    
    # Select the product
    df = data_df.loc[product]
    
    # Load the bands
    bands = dict()
    masks = dict()
    robust_scalers99 = dict()
    robust_scalers95 = dict()
    for i, r in df.iterrows():
        if r['bands_name'] in ['Red', 'Green', 'Blue']:
            # Load the band and mask
            with rasterio.open(i, 'r') as src:
                band = src.read()
                mask = src.read_masks()
                dtype = src.profile['dtype']
                profile = src.profile
            
            bands[r['bands_name']] = band
            # Add check for CBERS2 and Landsat, their masks are not reliable
            if product not in ['CBERS_2_CCD1XS_20081230_156_116', 'CBERS_2B_CCD1XS_20090905_167_120', 'LO82240772021365CUB00']:
                masks[r['bands_name']] = mask
            else:
                masks[r['bands_name']] = (band!=0).astype(np.uint8)
                
            # Train the RobustScalers
            robust_scalers99[r['bands_name']] = RobustScaler(quantile_range=(0, 99)).fit(band[mask.astype(np.bool)].reshape(-1, 1))
            band_norm = robust_scalers99[r['bands_name']].transform(band[mask.astype(np.bool)].reshape(-1, 1))
            robust_scalers99[f"{r['bands_name']}_min"], robust_scalers99[f"{r['bands_name']}_max"] = band_norm.min(), band_norm.max()
            robust_scalers95[r['bands_name']] = RobustScaler(quantile_range=(0, 95)).fit(band[mask.astype(np.bool)].reshape(-1, 1))
            band_norm = robust_scalers95[r['bands_name']].transform(band[mask.astype(np.bool)].reshape(-1, 1))
            robust_scalers95[f"{r['bands_name']}_min"], robust_scalers95[f"{r['bands_name']}_max"] = band_norm.min(), band_norm.max()
                
        elif r['bands_name'] in ['RGBN', 'RGBReN']:
            
            # Load bands
            with rasterio.open(i, 'r') as src:
                raster = src.read()
                dtype = src.profile['dtype']
                profile = src.profile
            # Add the bands to the bands dictionary
            bands['Red'] = raster[2][np.newaxis, :, :]
            bands['Green'] = raster[1][np.newaxis, :, :]
            bands['Blue'] = raster[0][np.newaxis, :, :]
            
            # Load masks
            if 'RapidEye' in i:
                i = i.replace('Analytic_SR', 'udm')
                with rasterio.open(i, 'r') as src:
                    masks_raster = src.read()
                masks_raster[masks_raster==2] = 0  # fix the cloud pixels
                masks_raster[masks_raster==0] = 255  # revert the mask
                masks_raster[masks_raster!=255] = 0  # change the other pixels to 0
                masks['Red'] = masks['Green'] = masks['Blue'] = masks_raster.copy()
            else:
                i = i.replace('BGRN_SR', 'udm2')
                with rasterio.open(i, 'r') as src:
                    masks_raster = src.read()
                final_mask = masks_raster[0]+masks_raster[2]+masks_raster[5]
                masks['Red'] = masks['Green'] = masks['Blue'] = final_mask[np.newaxis, :, :]
                
            # Train the RobustScalers
            for color in ['Red', 'Green', 'Blue']:
                robust_scalers99[color] = RobustScaler(quantile_range=(0, 99)).fit(bands[color][masks[color].astype(np.bool)].reshape(-1, 1))
                band_norm = robust_scalers99[color].transform(bands[color][masks[color].astype(np.bool)].reshape(-1, 1))
                robust_scalers99[f"{color}_min"], robust_scalers99[f"{color}_max"] = band_norm.min(), band_norm.max()
                robust_scalers95[color] = RobustScaler(quantile_range=(0, 95)).fit(bands[color][masks[color].astype(np.bool)].reshape(-1, 1))
                band_norm = robust_scalers95[color].transform(bands[color][masks[color].astype(np.bool)].reshape(-1, 1))
                robust_scalers95[f"{color}_min"], robust_scalers95[f"{color}_max"] = band_norm.min(), band_norm.max()
    
    # Train the mean band quantile scaler
    robust_scalers99['Mean'] = RobustScaler(quantile_range=(0, 99)).fit(np.mean(np.concatenate([bands['Red'], bands['Green'], bands['Blue']]), axis=0).reshape(-1, 1))
    robust_scalers99['Mean_min'], robust_scalers99['Mean_max'] = robust_scalers99['Mean'].transform(np.mean(np.concatenate([bands['Red'], bands['Green'], bands['Blue']]), axis=0).reshape(-1, 1)).min(), robust_scalers99['Mean'].transform(np.mean(np.concatenate([bands['Red'], bands['Green'], bands['Blue']]), axis=0).reshape(-1, 1)).max()
    robust_scalers95['Mean'] = RobustScaler(quantile_range=(0, 95)).fit(np.mean(np.concatenate([bands['Red'], bands['Green'], bands['Blue']]), axis=0).reshape(-1, 1))
    robust_scalers95['Mean_min'], robust_scalers95['Mean_max'] = robust_scalers95['Mean'].transform(np.mean(np.concatenate([bands['Red'], bands['Green'], bands['Blue']]), axis=0).reshape(-1, 1)).min(), robust_scalers95['Mean'].transform(np.mean(np.concatenate([bands['Red'], bands['Green'], bands['Blue']]), axis=0).reshape(-1, 1)).max()
    
    # Plot histograms for patches bands with the trained Scalers
    patches = extract_patches(raster=np.concatenate([bands['Red'], bands['Green'], bands['Blue']]),
                              mask=np.concatenate([masks['Red'], masks['Green'], masks['Blue']]), dim=dim)
    patches = np.moveaxis(patches, 0, 1)
    fig, axs = plt.subplots(2, 4, figsize=(48, 24))
    colors = {'Red': 'r', 'Green': 'g', 'Blue': 'b'}
    colors_idx = {'Green': 1, 'Red': 0, 'Blue': 2}
    for i, color in enumerate(['Green', 'Red', 'Blue']):
        norm_patches = robust_scalers99[color].transform(patches[colors_idx[color]].reshape(-1, 1)).reshape(patches[colors_idx[color]].shape)
        norm_patches = (norm_patches-robust_scalers99[f'{color}_min'])/(robust_scalers99[f'{color}_max']-robust_scalers99[f'{color}_min'])
        axs[0][0].hist(norm_patches.flatten(), 
                       bins=200, range=(0, 1), color=colors[color], 
                       label=color, alpha=0.8, density=True)
        axs[0][0].set_title('MinQMax (0-99th quantile) patches scaling distribution', fontsize=fsize)
        axs[0][0].tick_params(axis='both', labelsize=fsize)
        axs[0][0].legend(fontsize=fsize)
        norm_patches = robust_scalers99['Mean'].transform(patches[colors_idx[color]].reshape(-1, 1)).reshape(patches[colors_idx[color]].shape)
        norm_patches = (norm_patches-robust_scalers99[f'Mean_min'])/(robust_scalers99[f'Mean_max']-robust_scalers99[f'Mean_min'])
        axs[0][1].hist(norm_patches.flatten(), 
                    bins=200, range=(0, 1), color=colors[color], 
                    label=color, alpha=0.8, density=True)
        axs[0][1].set_title('MeanMinQMax (0-99th quantile) patches scaling distribution', fontsize=fsize)
        axs[0][1].tick_params(axis='both', labelsize=fsize)
        axs[0][1].legend(fontsize=fsize)
        norm_patches = robust_scalers95[color].transform(patches[colors_idx[color]].reshape(-1, 1)).reshape(patches[colors_idx[color]].shape)
        norm_patches = (norm_patches-robust_scalers95[f'{color}_min'])/(robust_scalers95[f'{color}_max']-robust_scalers95[f'{color}_min'])
        axs[0][2].hist(norm_patches.flatten(), 
                    bins=200, range=(0, 1), color=colors[color], 
                    label=color, alpha=0.8, density=True)
        axs[0][2].set_title('MinQMax (0-95th quantile) patches scaling distribution', fontsize=fsize)
        axs[0][2].tick_params(axis='both', labelsize=fsize)
        axs[0][2].legend(fontsize=fsize)
        norm_patches = robust_scalers95['Mean'].transform(patches[colors_idx[color]].reshape(-1, 1)).reshape(patches[colors_idx[color]].shape)
        norm_patches = (norm_patches-robust_scalers95[f'Mean_min'])/(robust_scalers95[f'Mean_max']-robust_scalers95[f'Mean_min'])
        axs[0][3].hist(norm_patches.flatten(), 
                    bins=200, range=(0, 1), color=colors[color], 
                    label=color, alpha=0.8, density=True)
        axs[0][3].set_title('MeanMinQMax (0-95th quantile) patches scaling distribution', fontsize=fsize)
        axs[0][3].tick_params(axis='both', labelsize=fsize)
        axs[0][3].legend(fontsize=fsize)
    

    # Let's reload the patches and plot them to see if the histogram matches with the previous ones
    product_save_path = os.path.join(save_path, product)
    scaler99 = load(os.path.join(product_save_path, '99_SatProductRobustScaler.joblib'))
    scaler95 = load(os.path.join(product_save_path, '95_SatProductRobustScaler.joblib'))
    patches_paths = sorted(glob.glob(os.path.join(product_save_path, '*_patch.tiff')))
    patches = []
    for path in patches_paths:
        with rasterio.open(path, 'r') as src:
            patches.append(src.read())
    patches = np.moveaxis(np.array(patches), 0, 1)
    for i, color in enumerate(['Green', 'Red', 'Blue']):
        norm_patches = scaler99.scalers[color].transform(patches[colors_idx[color]].reshape(-1, 1)).reshape(patches[colors_idx[color]].shape)
        norm_patches = (norm_patches-scaler99.min[color])/(scaler99.max[color]-scaler99.min[color])
        axs[1][0].hist(norm_patches.flatten(), 
                       bins=200, range=(0, 1), color=colors[color], 
                       label=color, alpha=0.8, density=True)
        axs[1][0].set_title('MinQMax (0-99th quantile) patches scaling distribution', fontsize=fsize)
        axs[1][0].tick_params(axis='both', labelsize=fsize)
        axs[1][0].legend(fontsize=fsize)
        norm_patches = scaler99.scalers['Mean'].transform(patches[colors_idx[color]].reshape(-1, 1)).reshape(patches[colors_idx[color]].shape)
        norm_patches = (norm_patches-scaler99.min['Mean'])/(scaler99.max['Mean']-scaler99.min['Mean'])
        axs[1][1].hist(norm_patches.flatten(), 
                    bins=200, range=(0, 1), color=colors[color], 
                    label=color, alpha=0.8, density=True)
        axs[1][1].set_title('MeanMinQMax (0-99th quantile) patches scaling distribution', fontsize=fsize)
        axs[1][1].tick_params(axis='both', labelsize=fsize)
        axs[1][1].legend(fontsize=fsize)
        norm_patches = scaler95.scalers[color].transform(patches[colors_idx[color]].reshape(-1, 1)).reshape(patches[colors_idx[color]].shape)
        norm_patches = (norm_patches-scaler95.min[color])/(scaler95.max[color]-scaler95.min[color])
        axs[1][2].hist(norm_patches.flatten(), 
                    bins=200, range=(0, 1), color=colors[color], 
                    label=color, alpha=0.8, density=True)
        axs[1][2].set_title('MinQMax (0-95th quantile) patches scaling distribution', fontsize=fsize)
        axs[1][2].tick_params(axis='both', labelsize=fsize)
        axs[1][2].legend(fontsize=fsize)
        norm_patches = scaler95.scalers['Mean'].transform(patches[colors_idx[color]].reshape(-1, 1)).reshape(patches[colors_idx[color]].shape)
        norm_patches = (norm_patches-scaler95.min['Mean'])/(scaler95.max['Mean']-scaler95.min['Mean'])
        axs[1][3].hist(norm_patches.flatten(), 
                    bins=200, range=(0, 1), color=colors[color], 
                    label=color, alpha=0.8, density=True)
        axs[1][3].set_title('MeanMinQMax (0-95th quantile) patches scaling distribution', fontsize=fsize)
        axs[1][3].tick_params(axis='both', labelsize=fsize)
        axs[1][3].legend(fontsize=fsize)
    fig.suptitle(f"{product} bands and PMF, {dtype} bits depth", fontsize=fsize)
    plt.show()


# ### Let's divide the patches in training and testing
# We'll use simple symbolic links.  
# We will not use for the moment:
# 1. S2A_MSIL2A_20220912T081611_N0400_R121_T35LLL_20220912T125758;
# 2. CBERS_4A_WFI_20220905_223_124;
# 3. all CBERS2 products.

# In[35]:


from sklearn.model_selection import train_test_split

save_path = '/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/resized_patches'
for product in data_df.drop(['S2A_MSIL2A_20220912T081611_N0400_R121_T35LLL_20220912T125758',
              'CBERS_4A_WFI_20220905_223_124', 'CBERS_2B_CCD1XS_20090905_167_120',
              'CBERS_2_CCD1XS_20081230_156_116', 'CBERS_4_PAN10M_20220905_151_116']).index.get_level_values(0).unique():
    if product not in ['train_patches', 'val_patches', 'test_patches']:
        # Create directories
        os.makedirs(os.path.join(save_path, 'train_patches', product), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'val_patches', product), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'test_patches', product), exist_ok=True)
        
        # Create the splits
        all_patches = glob.glob(os.path.join(save_path, product, '*.tiff'))
        test_split, trainval_split = train_test_split(all_patches, test_size=0.75, random_state=np.random.seed(42))
        val_split, train_split = train_test_split(trainval_split, test_size=0.50, random_state=np.random.seed(42))
        print(f'Product {product} - Train split: {len(train_split)} - Val split : {len(val_split)} - Test split: {len(test_split)}')
        
        # Create the symlinks
        for path in train_split:
            os.symlink(path, os.path.join(save_path, 'train_patches', product, os.path.split(path)[-1]))
        for path in val_split:
            os.symlink(path, os.path.join(save_path, 'val_patches', product, os.path.split(path)[-1]))
        for path in test_split:
            os.symlink(path, os.path.join(save_path, 'test_patches', product, os.path.split(path)[-1]))
        
        # Let's just check that we've done everything good
        train_patches = glob.glob(os.path.join(save_path, 'train_patches', product, '*.tiff'))
        val_patches = glob.glob(os.path.join(save_path, 'val_patches', product, '*.tiff'))
        test_patches = glob.glob(os.path.join(save_path,'test_patches', product, '*.tiff'))
        print(f'Train split is correct? {len(train_split)==len(train_patches)}\nVal split is correct? {len(val_split)==len(val_patches)}\nTrain split is correct? {len(test_split)==len(test_patches)}')
        
        # Copy also the ProductScalers
        all_scalers = glob.glob(os.path.join(save_path, product, '*.joblib'))
        for scaler in all_scalers:
            os.symlink(scaler, os.path.join(save_path, 'train_patches', product, os.path.split(scaler)[-1]))
            os.symlink(scaler, os.path.join(save_path, 'val_patches', product, os.path.split(scaler)[-1]))
            os.symlink(scaler, os.path.join(save_path, 'test_patches', product, os.path.split(scaler)[-1]))


# ### Copy only the scalers

# In[16]:


from sklearn.model_selection import train_test_split

save_path = '/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/patches'
for product in data_df.drop(['S2A_MSIL2A_20220912T081611_N0400_R121_T35LLL_20220912T125758',
              'CBERS_4A_WFI_20220905_223_124', 'CBERS_2B_CCD1XS_20090905_167_120',
              'CBERS_2_CCD1XS_20081230_156_116', 'CBERS_4_PAN10M_20220905_151_116']).index.get_level_values(0).unique():
    if product not in ['train_patches', 'val_patches', 'test_patches']:        
        # Copy also the ProductScalers
        scaler = os.path.join(save_path, product, 'SatProductMaxScaler.joblib')
        
        os.symlink(scaler, os.path.join(save_path, 'train_patches', product, os.path.split(scaler)[-1]))
        os.symlink(scaler, os.path.join(save_path, 'val_patches', product, os.path.split(scaler)[-1]))
        os.symlink(scaler, os.path.join(save_path, 'test_patches', product, os.path.split(scaler)[-1]))


# In[15]:


scaler


# ## DATA AUGMENTATION
# Let's apply some data augmentation to the patches

# ### Resizing

# In[19]:


from sklearn.model_selection import train_test_split
import cv2

source_dir = '/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/patches'
save_dir = '/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/resized_patches'

for product in data_df.drop(['S2A_MSIL2A_20220912T081611_N0400_R121_T35LLL_20220912T125758',
              'CBERS_4A_WFI_20220905_223_124', 'CBERS_2B_CCD1XS_20090905_167_120',
              'CBERS_2_CCD1XS_20081230_156_116', 'CBERS_4_PAN10M_20220905_151_116']).index.get_level_values(0).unique():
# for product in ['CBERS_4A_WFI_20220908_210_132']:
    if product not in ['train_patches', 'val_patches', 'test_patches']:
        # Create directory for saving
        os.makedirs(os.path.join(save_dir, product), exist_ok=True)
        
        # Look for all the patches
        all_files = glob.glob(os.path.join(source_dir, product, '*.tiff'))
        #all_files = all_files[:5]
        
        # Cycle for opening the file, augmenting it, and cropping it again
#         for path in all_files:
#             fig, axs = plt.subplots(1, 3, figsize=(24, 12))
#             with rasterio.open(path, 'r') as src:
#                 img = src.read()
#                 profile = src.profile
#             img = np.moveaxis(img, 0, 2)
#             axs[0].imshow(quantile_scaling(img, 'uniform'))
#             axs[1].imshow(quantile_scaling(img, 'uniform')[128-64:128+64, 128-64:128+64])
#             img_height = int(img.shape[0]*1.5)
#             img_width = int(img.shape[1]*1.5)
#             img = cv2.resize(img, (img_width, img_height))
#             print(f'Resized img shape is {img.shape}, dtype {img.dtype}')
#             img = img[int(img_height/2)-128:int(img_height/2)+128, int(img_width/2)-128:int(img_width/2)+128, :]
#             print(f'Final img shape is {img.shape}, dtype {img.dtype}')
#             axs[2].imshow(quantile_scaling(img, 'uniform'))
        for path in all_files:
            # open file
            with rasterio.open(path, 'r') as src:
                img = src.read()
                profile = src.profile
            img = np.moveaxis(img, 0, 2)  # move channel axis as last
            # Get resizing parameters
            img_height = int(img.shape[0]*1.5)
            img_width = int(img.shape[1]*1.5)
            # Resize
            img = cv2.resize(img, (img_width, img_height))
            # Center crop
            img = img[int(img_height/2)-128:int(img_height/2)+128, int(img_width/2)-128:int(img_width/2)+128, :]
            img = np.moveaxis(img, 2, 0)  # replace axis in place
            # Save the image
            filename = path.split('/')[-1]
            with rasterio.open(os.path.join(save_dir, product, filename), 'w', **profile) as dst:
                dst.write(img)        


# In[20]:


# Copy also the Scalers
import shutil

source_dir = '/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/patches'
save_dir = '/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/resized_patches'
for product in data_df.drop(['S2A_MSIL2A_20220912T081611_N0400_R121_T35LLL_20220912T125758',
              'CBERS_4A_WFI_20220905_223_124', 'CBERS_2B_CCD1XS_20090905_167_120',
              'CBERS_2_CCD1XS_20081230_156_116', 'CBERS_4_PAN10M_20220905_151_116']).index.get_level_values(0).unique():
# for product in ['CBERS_4A_WFI_20220908_210_132']:
    if product not in ['train_patches', 'val_patches', 'test_patches']:
        
        # Look for all the scalers
        all_files = glob.glob(os.path.join(source_dir, product, '*.joblib'))
        #all_files = all_files[:5]
        
        for file in all_files:
            shutil.copy(file, os.path.join(save_dir, product, file.split('/')[-1]))


# ### Let's make some splits using symlinks

# In[21]:


from sklearn.model_selection import train_test_split

save_path = '/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/resized_patches'
source_dir = '/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/patches'
for product in data_df.drop(['S2A_MSIL2A_20220912T081611_N0400_R121_T35LLL_20220912T125758',
              'CBERS_4A_WFI_20220905_223_124', 'CBERS_2B_CCD1XS_20090905_167_120',
              'CBERS_2_CCD1XS_20081230_156_116', 'CBERS_4_PAN10M_20220905_151_116']).index.get_level_values(0).unique():
    if product not in ['train_patches', 'val_patches', 'test_patches']:
        # Create directories
        os.makedirs(os.path.join(save_path, 'train_patches', product), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'val_patches', product), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'test_patches', product), exist_ok=True)
        
        # Create the splits
        all_patches = glob.glob(os.path.join(save_path, product, '*.tiff'))
        test_split, trainval_split = train_test_split(all_patches, test_size=0.75, random_state=np.random.seed(42))
        val_split, train_split = train_test_split(trainval_split, test_size=0.50, random_state=np.random.seed(42))
        print(f'Product {product} - Train split: {len(train_split)} - Val split : {len(val_split)} - Test split: {len(test_split)}')
        
        # Create the symlinks
        for path in train_split:
            os.symlink(path, os.path.join(save_path, 'train_patches', product, os.path.split(path)[-1]))
        for path in val_split:
            os.symlink(path, os.path.join(save_path, 'val_patches', product, os.path.split(path)[-1]))
        for path in test_split:
            os.symlink(path, os.path.join(save_path, 'test_patches', product, os.path.split(path)[-1]))
        
        # Let's just check that we've done everything good
        train_patches = glob.glob(os.path.join(save_path, 'train_patches', product, '*.tiff'))
        val_patches = glob.glob(os.path.join(save_path, 'val_patches', product, '*.tiff'))
        test_patches = glob.glob(os.path.join(save_path,'test_patches', product, '*.tiff'))
        print(f'Train split is correct? {len(train_split)==len(train_patches)}\nVal split is correct? {len(val_split)==len(val_patches)}\nTrain split is correct? {len(test_split)==len(test_patches)}')
        
        # Copy also the ProductScalers
        all_scalers = glob.glob(os.path.join(save_path, product, '*.joblib'))
        for scaler in all_scalers:
            os.symlink(scaler, os.path.join(save_path, 'train_patches', product, os.path.split(scaler)[-1]))
            os.symlink(scaler, os.path.join(save_path, 'val_patches', product, os.path.split(scaler)[-1]))
            os.symlink(scaler, os.path.join(save_path, 'test_patches', product, os.path.split(scaler)[-1]))


# ## Let's make some datasets for training

# ### The first dataset will account for tiles with and without resizing augmentation

# In[22]:


datasets_dir = '/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/train_datasets/resizing_augmentation'


# #### TRAIN SPLIT

# In[24]:


# TRAIN SPLIT
train_split_dir = os.path.join(datasets_dir, 'train_split')
os.makedirs(train_split_dir, exist_ok=True)

# Source dirs
source_dir = '/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/patches/train_patches'
source_dir_resized = '/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/resized_patches/train_patches'

# First non-resized patches
for product in os.listdir(source_dir):
    os.symlink(os.path.join(source_dir, product), os.path.join(train_split_dir, product), target_is_directory=True)

# Then resized patches
for product in os.listdir(source_dir_resized):
    os.symlink(os.path.join(source_dir_resized, product), os.path.join(train_split_dir, f"{product}_resized"), target_is_directory=True)


# Let's see some images

# In[25]:


for product in os.listdir(train_split_dir):
    image_path = sorted(glob.glob(os.path.join(train_split_dir, product, '*.tiff')))[0]
    with rasterio.open(image_path, 'r') as src:
        img = src.read()
    plt.imshow(quantile_scaling(np.moveaxis(img, 0, 2), 'uniform'))
    plt.title(f'Example of product {product}')
    plt.show()


# #### VAL SPLIT

# In[26]:


# TRAIN SPLIT
split = 'val'
split_dir = os.path.join(datasets_dir, f'{split}_split')
os.makedirs(split_dir, exist_ok=True)

# Source dirs
source_dir = f'/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/patches/{split}_patches'
source_dir_resized = f'/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/resized_patches/{split}_patches'

# First non-resized patches
for product in os.listdir(source_dir):
    os.symlink(os.path.join(source_dir, product), os.path.join(split_dir, product), target_is_directory=True)

# Then resized patches
for product in os.listdir(source_dir_resized):
    os.symlink(os.path.join(source_dir_resized, product), os.path.join(split_dir, f"{product}_resized"), target_is_directory=True)


# In[27]:


for product in os.listdir(split_dir):
    image_path = sorted(glob.glob(os.path.join(split_dir, product, '*.tiff')))[0]
    with rasterio.open(image_path, 'r') as src:
        img = src.read()
    plt.imshow(quantile_scaling(np.moveaxis(img, 0, 2), 'uniform'))
    plt.title(f'Example of product {product}')
    plt.show()


# In[12]:


os.path.exists('/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/train_datasets/resizing_augmentation/train_split/AMAZONIA_1_WFI_20220908_035_018_resized/95_SatProductRobustScaler.joblib')


# In[ ]:




