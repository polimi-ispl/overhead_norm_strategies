#!/usr/bin/env bash

# USER PARAMETERS (put your device configuration params here)
DEVICE=0
SGD_DIR=../data/spliced_images/sgd # path to SGD dataset (YOUR PATH MAY BE DIFFERENT IF YOU SAVED THE SPLITS IN ANOTHER FOLDER, CHECK IT)
HEGD_DIR=../data/spliced_images/hegd # path to HEGD dataset (YOUR PATH MAY BE DIFFERENT IF YOU SAVED THE SPLITS IN ANOTHER FOLDER, CHECK IT)
RESULTS_DIR=../results/spliced_images # path to save the results (YOUR PATH MAY BE DIFFERENT IF YOU SAVED THE SPLITS IN ANOTHER FOLDER, CHECK IT)
PREPROCESSING_DIR=../data/pristine_images/patches # path to the folder with the original patches (needed for retrieving the MinPMax scaler. YOUR PATH MAY BE DIFFERENT IF YOU SAVED THE SPLITS IN ANOTHER FOLDER, CHECK IT)

echo ""
echo "-------------------------------------------------"
echo "| Test with MinPMax 99th percentile threshold |"
echo "-------------------------------------------------"
echo "| SGD dataset |"
python ../test_with_AUCs.py --gpu $DEVICE --fe_path ..models/MinPMax_99/model_weights.h5 \
--root_dir $SGD_DIR --results_dir $RESULTS_DIR --preprocessing_dir $PREPROCESSING_DIR
echo "| HEGD dataset |"
python ../test_with_AUCs.py --gpu $DEVICE --fe_path ..models/MinPMax_99/model_weights.h5 \
--root_dir $HEGD_DIR --results_dir $RESULTS_DIR --preprocessing_dir $PREPROCESSING_DIR


echo ""
echo "-------------------------------------------------"
echo "| Test with MinPMax 95th percentile threshold |"
echo "-------------------------------------------------"
echo "| SGD dataset |"
python ../test_with_AUCs.py --gpu $DEVICE --fe_path ..models/MinPMax_95/model_weights.h5 \
--root_dir $SGD_DIR --results_dir $RESULTS_DIR --preprocessing_dir $PREPROCESSING_DIR
echo "| HEGD dataset |"
python ../test_with_AUCs.py --gpu $DEVICE --fe_path ..models/MinPMax_95/model_weights.h5 \
--root_dir $HEGD_DIR --results_dir $RESULTS_DIR --preprocessing_dir $PREPROCESSING_DIR

echo ""
echo "-------------------------------------------------"
echo "| Test with MaxAbs scaling |"
echo "-------------------------------------------------"
echo "| SGD dataset |"
python ../test_with_AUCs.py --gpu $DEVICE --fe_path ..models/MaxScaling/model_weights.h5 \
--root_dir $SGD_DIR --results_dir $RESULTS_DIR
echo "| HEGD dataset |"
python ../test_with_AUCs.py --gpu $DEVICE --fe_path ..models/MaxScaling/model_weights.h5 \
--root_dir $HEGD_DIR --results_dir $RESULTS_DIR

echo ""
echo "-------------------------------------------------"
echo "| Test with HistogramEqualization scaling |"
echo "-------------------------------------------------"
echo "| SGD dataset |"
python ../test_with_AUCs.py --gpu $DEVICE --fe_path ..models/HistogramEqualization/model_weights.h5 \
--root_dir $SGD_DIR --results_dir $RESULTS_DIR
echo "| HEGD dataset |"
python ../test_with_AUCs.py --gpu $DEVICE --fe_path ..models/HistogramEqualization/model_weights.h5 \
--root_dir $HEGD_DIR --results_dir $RESULTS_DIR