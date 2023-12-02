"""
Test with AUCs

A simple script to compute the fingerprints, binarize them, and compute the ROC curves and AUCs.
Note: the models outputs a fingerprint for each channel. We simply average them to get a single fingerprint.
Further research can be directed into using the fingerprints from each channel separately and/or combining them.

Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
Sriram Baireddy - sbairedd@purdue.edu
Paolo Bestagini - paolo.bestagini@polimi.it
Stefano Tubaro - stefano.tubaro@polimi.it
Edward J. Delp - ace@purdue.edu
"""

# Libraries import
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import keras
import os
import sys
from isplutils.network import generate_separable_fingerprint_extractor
from isplutils.data import Scaler, SatTilesScaler, SatProductRobustScaler, SatProductMaxScaler
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from tqdm import tqdm
import argparse
from multiprocessing import cpu_count
from numba import cuda
from joblib import load
import rasterio
import cv2


# Helper functions and classes
def get_input_scaler(fe_tag: str, data_df: pd.DataFrame, preprocesser_dir: str) -> pd.DataFrame:
    """
    Find and load the type of scaler from the model path
    """
    if 'MinPMax' in fe_tag:
        percentile_num = fe_tag.split('MinPMax')[-1].split('_')[1]
        data_df['target_scaler'] = data_df['target_satellite'].apply(
            lambda x: os.path.join(preprocesser_dir, x, f'{percentile_num}_SatProductRobustScaler.joblib'))
    elif fe_tag == 'MaxScaling':
        data_df['target_scaler'] = 'max_scaling'
    else:
        data_df['target_scaler'] = 'uniform_scaling'

    return data_df


def main(args: argparse.Namespace) -> None:

    # -- Load the DataFrame with the info on the test set -- #
    root_dir = args.root_dir
    spliced_df = pd.read_csv(os.path.join(root_dir, 'all_ops_df.csv'))

    # -- Load the fingerprint extractor -- #
    fe_path = args.fe_path
    gpu = args.gpu

    # GPU configuration
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)  # set the GPU device
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print('tf    version:', tf.__version__)
    print('keras version:', keras.__version__)

    configSess = tf.ConfigProto()
    # Allowing the GPU memory to grow avoiding preallocating all the GPU memory
    configSess.gpu_options.allow_growth = True
    set_session(tf.Session(config=configSess))

    # Load the actual model
    fe = generate_separable_fingerprint_extractor(output_channels=3, image_channels=3)
    fe.load_weights(fe_path)

    # Parse useful fields from the model's directory name
    fe_tag = fe_path.split('/')[-2]

    # -- Prepare the results dir and results DataFrame -- #
    results_dir = os.path.join(args.results_dir, args.root_dir.split('/')[-2], fe_tag,
                               f'unknown_satellite-{args.unknown_target}')
    os.makedirs(results_dir, exist_ok=True)
    results_df = spliced_df.copy()
    results_df = get_input_scaler(fe_tag, results_df, preprocesser_dir=args.preprocessing_dir)
    # Add results columns to the DataFrame
    results_df['AUC'] = np.nan
    results_df['fpr'] = ''
    results_df['tpr'] = ''

    # -- Compute everything -- #

    # Select a subsample of rows for debugging
    if args.debug:
        results_df = results_df.sample(n=10, random_state=42)

    # Load satellite product scalers
    fp_scalers = dict()
    for scaler_path in results_df['target_scaler'].unique():
        if os.path.exists(scaler_path):
            fp_scalers[scaler_path.split('/')[-2]] = load(scaler_path)
        else:
            fp_scalers[scaler_path] = SatTilesScaler()

    # Compute the fingerprints first, so we can free up the GPU memory when not needed
    for i, r in tqdm(results_df.iterrows()):
        # Load and normalize test image
        with rasterio.open(os.path.join(root_dir, r['img_path']), 'r') as src:
            img = src.read()
        if 'SatProduct' in r['target_scaler']:
            if not args.unknown_target:
                img = fp_scalers[r['target_satellite']].normalize_product(img, True)
            else:
                img = fp_scalers[r['target_satellite']].normalize_unknown_product(img, True)
        else:
            img = fp_scalers[scaler_path].normalize_product(img, scaler_path, True)

        # Compute the fingerprint
        fp = fe.predict(img[np.newaxis, :, :, :])

        # Load the GT mask
        mask = cv2.imread(os.path.join(root_dir, r['mask_path']), cv2.IMREAD_UNCHANGED)

        # Compute the ROC curves and AUCs

        # Average the fingerprints
        fp = np.mean(fp, axis=-1, keepdims=True)
        # Compute TPR, FPR, AUC
        fpr, tpr, _ = roc_curve(mask[:, :, 0].flatten(), fp.flatten(), pos_label=255)
        auc_score = auc(fpr, tpr)
        results_df.at[i, 'AUC'] = auc_score if auc_score > 0.5 else 1 - auc_score
        # Save the fpr, tpr
        results_df.at[i, 'fpr'] = [fpr.tolist()]
        results_df.at[i, 'tpr'] = [tpr.tolist()]

    # Release memory
    # keras.backend.clear_session()
    # reset_keras(fe)
    # THE ONLY WAY I'VE FOUND FOR RELEASING MEMORY WITH TF 1.15
    device = cuda.get_current_device()
    device.reset()

    # -- Save the results DataFrame -- #
    if args.debug:
        results_df.to_csv(os.path.join(results_dir, f'results_df_mean_fingerprints_DEBUG.csv'),
                             index=True, header=True, index_label=False)
    else:
        results_df.to_csv(os.path.join(results_dir, f'results_df_mean_fingerprints.pkl'),
                             index=True, header=True, index_label=False)

    return None


# --- MAIN --- #
if __name__ == '__main__':

    # -- Introduce arguments -- #
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, help='GPU to use for extracting the fingerprints', default=0)
    parser.add_argument('--fe_path', type=str, help='Path to the model\'s weights',
                        default='models/MinPMax_99/model_weights.h5')
    parser.add_argument('--root_dir', type=str, help='Path to the directory with the test samples',
                        default='data/spliced_images/hegd')
    parser.add_argument('--results_dir', type=str, help='Directory where to story the results',
                        default='splicing_results/auc')
    parser.add_argument('--preprocessing_dir', type=str, help='Directory where all scalers are contained',
                        default='data/pristine_images/patches')
    parser.add_argument('--unknown_target', action='store_true', help='Whether we know the satellite of the target '
                                                                    'image or not. BEWARE: works only with RobustScalers!')
    parser.add_argument('--slack_user', type=str, help='Slack user to warn on the ISPL workspace', default='edo.cannas')
    parser.add_argument('--debug', action='store_true', help='Execute in debug mode or not')

    config = parser.parse_args()

    # -- Call main -- #
    print('Starting the computation of the multi bands ROC curves...')
    try:
        main(config)
    except Exception as e:
        print('Something happened! Error is {}'.format(e))
    print('Testing done! Bye!')

    # -- Exit -- #
    sys.exit()