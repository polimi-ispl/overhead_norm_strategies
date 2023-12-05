"""
Copy-paste panchromatic images creation script
This script creates "copy-paste" panchromatic images, where the pasted region undergoes some processing.
Images are divided in folders by geographical region captured and satellite that acquired the image
We will create 50 copy-paste images for each satellite, region and for each processing pipeline we consider.
We will use the Imagaug library (https://github.com/aleju/imgaug) for the editing operations
Operations considered (for this script):
-GaussianBlur
-AverageBlur
-MotionBlur
-Affine
-PiecewiseAffine
-PerspectiveAffine
-SigmoidContrast
-Contrast
-Identity

Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
Sriram Baireddy - sbairedd@purdue.edu
Paolo Bestagini - paolo.bestagini@polimi.it
Stefano Tubaro - stefano.tubaro@polimi.it
Edward J. Delp - ace@purdue.edu
"""

# Libraries import #
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7, 7)

import glob
import os
import sys
sys.path.append('../')
import cv2
import imgaug.augmenters as iaa
import numpy as np
import argparse
import pandas as pd
from isplutils.slack import ISPLSlack
from skimage.segmentation import quickshift
from skimage.measure import regionprops
from sklearn.preprocessing import QuantileTransformer
import rasterio


# Helpers functions #

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a 2D square gaussian kernel.
    :param size is the length of a side of the square
    :param fwhm is full-width-half-maximum, which
    can be thought of as an effective radius (the variance of the Gaussian bell).
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


def splice(target_img, source_img, augs=None, modality='copy-paste', blend=False, input_norm='uniform'):
    # Transform the data
    target_dtype = target_img.dtype
    if input_norm == 'uniform':
        # Transform the data using a uniform histogram equalization
        transform_target = QuantileTransformer(output_distribution='uniform', random_state=42).fit(
            np.mean(target_img, axis=2).reshape(-1, 1))
        target_img = transform_target.transform(target_img.reshape(-1, 1)).reshape(target_img.shape)
        transform_source = QuantileTransformer(output_distribution='uniform', random_state=42).fit(
            np.mean(source_img, axis=2).reshape(-1, 1))
        source_img = transform_source.transform(source_img.reshape(-1, 1)).reshape(source_img.shape)
    elif input_norm == 'max':
        # Transform the data in a 0-1 range dividing by the maximum value of the patch
        max_target = target_img.max()
        target_img = target_img.astype(float)
        target_img /= float(max_target)
        source_img = source_img.astype(float)
        source_img /= float(source_img.max())
    elif input_norm == 'max_abs':
        # Transform the data in a 0-1 range dividing by the maximum value of the datatype
        target_dtype = target_img.dtype
        target_img = target_img.astype(float)
        # Divide by the maximum value of the datatype
        target_img /= np.iinfo(target_dtype).max
        # Repeat for the source image
        source_dtype = source_img.dtype
        source_img = source_img.astype(float)
        # Divide by the maximum value of the datatype
        source_img /= np.iinfo(source_dtype).max

    # Segment the target image with quickshift
    seg_mask = quickshift(target_img, kernel_size=9, max_dist=50, ratio=0.5)

    # Create the spliced image variable
    spliced_img = target_img.copy()
    if modality == 'copy-paste':
        # Editing the source image, using an ImgAugmentation pipeline
        if augs is not None:
            source_img = augs.augment_image(source_img)
        # Select one of the segmented regions randomly
        target_mask = np.zeros_like(target_img)
        target_mask[seg_mask == np.random.choice(np.unique(seg_mask), 1)[0]] = 255
        target_mask = target_mask.astype(bool)
        spliced_img[target_mask] = source_img[target_mask].copy()
    else:
        # WATCH OUT! we need to avoid regions too big in the original segmentation map
        # We need to evaluate each candidate mask separately, and get rid of the masks with a target area too wide
        candidate_masks = []
        for mask_value in np.unique(seg_mask):  # evaluate each candidate segmentation mask separately
            mask = np.zeros_like(target_img).astype(np.uint8)
            mask[seg_mask == mask_value] = 255
            spliced_area = regionprops(mask[:, :, 0])[0]
            min_row, min_col, max_row, max_col = spliced_area['bbox']
            # if the segmented area is too big, we do not consider the mask
            # we evaluate the dimension of the target area by considering the associated bounding box
            if ((max_col - min_col) < 256) and ((max_row - min_row) < 256):
                candidate_masks.append(mask)
        # Select one of the segmented regions randomly
        source_mask = candidate_masks[np.random.choice(np.arange(0, len(candidate_masks)), 1)[0]]
        source_mask = source_mask[:, :, 0]  # we want a single channel mask
        # Crop the source area from the pixel region
        # We are going to use the coordinates of the bounding box around the spliced area
        spliced_area = regionprops(source_mask)[0]
        min_row, min_col, max_row, max_col = spliced_area['bbox']  # get the coordinates
        patch_mask = source_mask[min_row:max_row, min_col:max_col]  # crop the source area in the source mask
        patch = source_img[min_row:max_row, min_col:max_col].copy()  # copy the source area from the source sample
        # Execute the splicing attack #
        patch[~patch_mask.astype(bool)] = 0  # set to 0 the pixels not in the real source area
        # augment both patch and relative masks, in order to have the right ground-truth for distortion editing
        # (i.e., PerspectiveTransform)
        patch, patch_mask = op(image=patch, segmentation_maps=patch_mask[np.newaxis, :, :, np.newaxis])
        patch_mask = patch_mask[0, :, :, 0]
        # Find a random point where to place the edited area
        rows_target_img, cols_target_img, color_channels = np.shape(target_img)
        p_row = np.random.randint(0, rows_target_img - (max_row - min_row) - 1)
        p_col = np.random.randint(0, cols_target_img - (max_col - min_col) - 1)
        # Attach the area
        target_mask = np.zeros_like(source_mask)
        a, b = np.where(patch_mask > 0)  # find coordinates of the source area in the patch
        spliced_img[a + p_row, b + p_col] = 0  # set to zero the same pixels in corresponding target area
        # Execute the attack with a simple addition (pixel inside the target area will come from the source area,
        # and those outside are pristine from the target sample)
        spliced_img[p_row: p_row + (max_row - min_row), p_col: p_col + (max_col - min_col)] = \
            patch + spliced_img[p_row: p_row + (max_row - min_row), p_col: p_col + (max_col - min_col)]
        target_mask[p_row: p_row + (max_row - min_row), p_col: p_col + (max_col - min_col)] = patch_mask.copy()

    # Blend the pasted region with the original target image
    target_mask = target_mask.astype(np.uint8) * 255  # first convert to uint8 mask
    center = regionprops(target_mask)[0]['centroid'][1], regionprops(target_mask)[0]['centroid'][0]
    area = regionprops(target_mask)[0]['area']
    if blend:
        blend_mask = makeGaussian(spliced_img.shape[0], np.sqrt(area), center)[:, :, np.newaxis]
        spliced_img = np.multiply(blend_mask, spliced_img) + np.multiply((1 - blend_mask), target_img)
    else:
        blend_mask = target_mask

    # Re-transform back
    if input_norm == 'uniform':
        spliced_img = transform_target.inverse_transform(spliced_img.reshape(-1, 1)).reshape(spliced_img.shape).astype(
            target_dtype)
    elif input_norm == 'max':
        spliced_img *= max_target
    elif input_norm == 'max_abs':
        spliced_img *= np.iinfo(target_dtype).max

    return [spliced_img.astype(target_dtype), target_mask, blend_mask]


# Argument parsing #
parser = argparse.ArgumentParser()
parser.add_argument('--source_img_dir', type=str, help='Directory containing the tiles used for creating the dataset',
                    default='/home/edcannas/projects/rgb_fingerprint_extractor/data/pristine_images/patches/test_patches')
parser.add_argument('--split_seed', type=int, help='Seed for splitting images in sources and targets', default=42)
parser.add_argument('--destination_dir', help='Destination directory for storing the copy-paste images', required=True)
parser.add_argument('--num_imgs', type=int, default=50, help='Number of copy-paste images per editing operation, '
                                                             'satellite and region. The total # of generated images '
                                                             'will be equal '
                                                             '= num_imgs*num_editing_ops*num_satellites')
parser.add_argument('--splicing_mod', type=str, help='Splicing modality, either copy-paste (source satellite != target'
                                                     'satellite) or copy-move (source satellite == target satellite)',
                    default='copy-paste', choices=['copy-paste', 'copy-move'])
parser.add_argument('--blend', action='store_true', help='Whether to blend the tampered region using a Gaussian bell')
parser.add_argument('--input_norm', type=str, choices=['uniform', 'max', 'max_abs'],
                    help='Normalization strategy for the target and source samples in a copy-paste', default='uniform')
parser.add_argument('--slack_user', type=str, default='edo.cannas', help='Slack user to which send a message')
parser.add_argument('--same_bit_depth', action='store_true', help='Whether to create copy-paste between images of the'
                                                                  'same bit resolution')
parser.add_argument('--debug', action='store_true', help='Debug flag')


### MAIN ###
if __name__ == '__main__':
    print('Starting...')

    # --- Argument parsing --- #
    args = parser.parse_args()
    debug = args.debug
    source_img_dir = args.source_img_dir
    split_seed = args.split_seed
    dest_dir = args.destination_dir
    slack_user = args.slack_user
    modality = args.splicing_mod
    blend = args.blend
    input_norm = args.input_norm
    same_bit_depth = args.same_bit_depth
    if debug:
        dest_dir = os.path.join(dest_dir, 'debug')
    if debug:
        num_imgs = 10
    else:
        num_imgs = args.num_imgs

    # --- Loading source DataFrame --- #
    print('Picking test images...')
    all_patches = glob.glob(os.path.join(source_img_dir, '**/*.tiff'), recursive=True)
    patches_df = pd.concat(
        [pd.concat({path.split('/')[-2]: pd.DataFrame(index=[path])}, names=['Original_product', 'patch_path']) for path
         in all_patches])
    patches_df['bit_depth'] = ''
    for i, r in patches_df.iterrows():
        with rasterio.open(i[1], 'r') as src:
            patches_df.loc[i, 'bit_depth'] = src.profile['dtype']
    # patches_df = pd.concat([patches_df], keys=patches_df['bit_depth'], names=['Bit_depth'])
    patches_df['product'] = patches_df.index.get_level_values(0)

    # --- Let's create the target DataFrame --- #

    # Prepare list of augmentations
    if debug:
        ops = [iaa.AverageBlur(k=((2, 10), (2, 10))),  # blur
               iaa.GaussianBlur(sigma=(0, 5)),
               iaa.Identity()  # identity transform, for simple copy-paste
              ]
    else:
        ops = [iaa.AverageBlur(k=((2, 10), (2, 10))),  # blur
               iaa.GaussianBlur(sigma=(0, 5)),
               iaa.MotionBlur(k=5),
               iaa.Affine(scale=(1, 1.5), rotate=(-90, 90), backend='cv2'),  # affine
               iaa.PiecewiseAffine(scale=(0.01, 0.3)),
               iaa.PerspectiveTransform(scale=(0.01, 0.15)),
               iaa.SigmoidContrast(gain=(5, 20), cutoff=(0.25, 0.75)),  # contrast
               iaa.LogContrast(gain=(0.6, 1.4)),
               iaa.Identity()  # identity transform, for simple copy-paste
               ]

    # Prepare results DataFrame
    num_imgs_per_op = num_imgs * len(patches_df.index.get_level_values(0).unique())
    results_df = pd.DataFrame(index=pd.MultiIndex.from_product(
        [[op.name for op in ops], ['img_{}.tiff'.format(idx) for idx in range(num_imgs_per_op)]]))
    results_df['img_path'] = np.NaN
    results_df['mask_path'] = np.NaN
    results_df['blend_mask_path'] = np.NaN
    results_df['target_satellite'] = np.NaN
    results_df['source_satellite'] = np.NaN
    results_df['spliced_area'] = np.NaN

    # --- Set Slack user --- #
    if slack_user is not None:
        slack_m = ISPLSlack()
        slack_m.to_user(recipient=slack_user, message='Starting splicing creation of RGB overhead images...')

    # --- MAIN LOOP --- #
    dest_dir = os.path.join(dest_dir, f'{modality}_blend-{blend}_input_norm-{input_norm}_same_bit_depth-{same_bit_depth}')
    # Loop over operations
    for op in ops:
        print('Doing operation {}...'.format(op.name))
        cnt = 0
        # Proceed by satellite and then by region
        for satellite in patches_df.index.get_level_values(0).unique():
            if modality == 'copy-paste':
                # Pick target and source images (TARGETS and SOURCES MUST be from different satellites!)
                target_imgs = patches_df.loc[satellite].sample(num_imgs)
                if same_bit_depth:
                    source_imgs = patches_df.loc[(~patches_df['product'].isin([satellite])) & (
                        patches_df['bit_depth'].isin(target_imgs['bit_depth'].unique()))].sample(num_imgs)
                else:
                    source_imgs = patches_df.loc[(~patches_df['product'].isin([satellite]))].sample(num_imgs)
                if np.any(source_imgs.index.get_level_values(0).unique().isin([satellite])):
                    raise RuntimeError('Something is going on here! Source and target sats are the same!')
                else:
                    print('Source and target sats are different, let\'s go!')
                source_imgs = source_imgs.droplevel(0)  # drop outer index, we do not need it anymore
            else:
                # Pick target and source images (TARGETS and SOURCES ARE from same satellite!)
                target_imgs = patches_df.loc[satellite].sample(num_imgs)
                source_imgs = target_imgs.copy()
            # Create the splicings
            for i, record in enumerate(target_imgs.iterrows()):
                # Splice
                idx, r = record
                # Load target raster
                with rasterio.open(idx, 'r') as src:
                    target = src.read()
                    target = np.moveaxis(target, 0, 2)
                    target_profile = src.profile
                # Load source raster
                with rasterio.open(source_imgs.iloc[i].name, 'r') as src:
                    source = src.read()
                    source = np.moveaxis(source, 0, 2)
                # Execute the splicing
                spliced, mask, blend_mask = splice(target_img=target, source_img=source, augs=op,
                                                   modality=modality, blend=blend, input_norm=input_norm)
                # Save images and masks
                save_dir = os.path.join(dest_dir, '{}'.format(op.name))
                os.makedirs(save_dir, exist_ok=True)
                with rasterio.open(os.path.join(save_dir, 'img_{}.tiff'.format(cnt)), 'w', **target_profile) as dst:
                    dst.write(np.moveaxis(spliced, 2, 0))
                cv2.imwrite(os.path.join(save_dir, 'mask_img_{}.png'.format(cnt)), mask)
                cv2.imwrite(os.path.join(save_dir, 'blend_mask_img_{}.png'.format(cnt)), blend_mask)
                # Update the results Dataframe
                results_df.loc[(op.name, 'img_{}.tiff'.format(cnt)), 'img_path'] = os.path.join(save_dir,
                                                                                                'img_{}.tiff'.format(
                                                                                                    cnt))
                results_df.loc[(op.name, 'img_{}.tiff'.format(cnt)), 'mask_path'] = os.path.join(save_dir,
                                                                                                 'mask_img_{}.png'.format(
                                                                                                     cnt))
                results_df.loc[(op.name, 'img_{}.tiff'.format(cnt)), 'blend_mask_path'] = os.path.join(save_dir,
                                                                                                       'blend_mask_img_{}.png'.format(
                                                                                                           cnt))
                results_df.loc[(op.name, 'img_{}.tiff'.format(cnt)), 'target_satellite'] = satellite
                results_df.loc[(op.name, 'img_{}.tiff'.format(cnt)), 'source_satellite'] = source_imgs.iloc[i].name.split('/')[-2]
                results_df.loc[(op.name, 'img_{}.tiff'.format(cnt)), 'spliced_area'] = np.sqrt(regionprops(mask)[0]['area'])
                # Update cnt for indexing
                cnt += 1

    # Done! Saving the DataFrame for navigation
    print('All operations done! Saving the result DataFrame...')
    if debug:
        results_df.to_pickle(os.path.join(dest_dir, 'all_ops_df_DEBUG.pkl'))
    else:
        results_df.to_pickle(os.path.join(dest_dir, 'all_ops_df.pkl'))

    if slack_user is not None:
        slack_m = ISPLSlack()
        slack_m.to_user(recipient=slack_user, message='Splicing creation finished!')