#!/bin/bash

# =================================================================
# CONFIGURATION
# =================================================================
# Input Data
CSV_PATH="data/pathology.csv"
TRAIN_CSV="data/pathology_train.csv"
VAL_CSV="data/pathology_val.csv"

# Directories
PATCH_DIR="processed_data/patches_output/patches"
IMAGE_DIR="processed_data/resized"
FEATURE_DIR="processed_data/features_pathology"
SAVE_DIR="checkpoints/pathology"

# Create output directories
mkdir -p $FEATURE_DIR
mkdir -p $SAVE_DIR

# =================================================================
# STEP 3: Train CNN on Pathology Labels
# =================================================================
echo "----------------------------------------------------------------"
echo "Starting STEP 3: Training CNN on Pathology Labels..."
echo "----------------------------------------------------------------"

python seattn_train/step3_train_pathology_cnn.py \
    --csv_path $CSV_PATH \
    --patch_folder $PATCH_DIR \
    --image_folder $IMAGE_DIR \
    --save_dir $SAVE_DIR \
    --gc 4

# =================================================================
# STEP 4: Extract Features using Trained CNN
# =================================================================
echo "----------------------------------------------------------------"
echo "Starting STEP 4: Extracting Features..."
echo "----------------------------------------------------------------"

python seattn_train/step4_extract_features.py \
    --csv_path $CSV_PATH \
    --patch_dir $PATCH_DIR \
    --image_folder $IMAGE_DIR \
    --model_path "$SAVE_DIR/phase1_cnn.pth" \
    --feature_dir $FEATURE_DIR

# =================================================================
# STEP 5: Train Attention on Extracted Features
# =================================================================
echo "----------------------------------------------------------------"
echo "Starting STEP 5: Training Attention Mechanism..."
echo "----------------------------------------------------------------"

python seattn_train/step5_train_pathology_attn.py \
    --train_csv $TRAIN_CSV \
    --val_csv $VAL_CSV \
    --feature_dir $FEATURE_DIR \
    --save_dir $SAVE_DIR \
    --epochs 50 \
    --gc 4

echo "================================================================"
echo "Pathology Pipeline Completed Successfully!"
echo "================================================================"