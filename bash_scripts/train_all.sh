#!/usr/bin/env bash

# USER PARAMETERS (put your device configuration params here)
DEVICE=0
TRAIN_DIR=../data/pristine_images/train_patches # path to training patches (YOUR PATH MAY BE DIFFERENT IF YOU SAVED THE SPLITS IN ANOTHER FOLDER, CHECK IT)
VAL_DIR=../data/pristine_images/val_patches # path to training patches (YOUR PATH MAY BE DIFFERENT IF YOU SAVED THE SPLITS IN ANOTHER FOLDER, CHECK IT)

echo ""
echo "-------------------------------------------------"
echo "| Train with MinPMax 99th percentile threshold |"
echo "-------------------------------------------------"
python ../train.py --gpu $DEVICE --batch_size 10 --num_iteration 128 --learning_rate 0.0001 \
--epochs 500 --train_dir $TRAIN_DIR --val_dir $VAL_DIR --num_tiles_peracq 200 --batch_num_num_tiles_peracq 10 \
--batch_num_pos_pertile 6 --scaler_type 99th_percentile --mean_robust_scaling --input_fp_channels 3 \
--output_fp_channels 3

echo ""
echo "-------------------------------------------------"
echo "| Train with MinPMax 95th percentile threshold |"
echo "-------------------------------------------------"
python ../train.py --gpu $DEVICE --batch_size 10 --num_iteration 128 --learning_rate 0.0001 \
--epochs 500 --train_dir $TRAIN_DIR --val_dir $VAL_DIR --num_tiles_peracq 200 --batch_num_num_tiles_peracq 10 \
--batch_num_pos_pertile 6 --scaler_type 95th_percentile --mean_robust_scaling --input_fp_channels 3 \
--output_fp_channels 3

echo ""
echo "-------------------------------------------------"
echo "| Train with MaxAbs scaling |"
echo "-------------------------------------------------"
python ../train.py --gpu $DEVICE --batch_size 10 --num_iteration 128 --learning_rate 0.0001 \
--epochs 500 --train_dir $TRAIN_DIR --val_dir $VAL_DIR --num_tiles_peracq 200 --batch_num_num_tiles_peracq 10 \
--batch_num_pos_pertile 6 --scaler_type sat_tiles_scaler --input_norm max_scaling --input_fp_channels 3 \
--output_fp_channels 3

echo ""
echo "-------------------------------------------------"
echo "| Train with HistogramEqualization scaling |"
echo "-------------------------------------------------"
python ../train.py --gpu $DEVICE --batch_size 10 --num_iteration 128 --learning_rate 0.0001 \
--epochs 500 --train_dir $TRAIN_DIR --val_dir $VAL_DIR --num_tiles_peracq 200 --batch_num_num_tiles_peracq 10 \
--batch_num_pos_pertile 6 --scaler_type sat_tiles_scaler --input_norm uniform_scaling --input_fp_channels 3 \
--output_fp_channels 3