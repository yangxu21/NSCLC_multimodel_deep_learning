#!/bin/bash

# =================================================================
# CONFIGURATION
# =================================================================
# Input Data
CSV_PATH="data/survival.csv"
TRAIN_CSV="data/survival_train.csv"
VAL_CSV="data/survival_val.csv"
PATHOLOGY_WEIGHTS="checkpoints/pathology/phase1_cnn.pth"

# Directories
PATCH_DIR="processed_data/patches_output/patches"
IMAGE_DIR="processed_data/resized"
FEATURE_DIR="processed_data/features_survival"
SAVE_DIR="checkpoints/survival"

# Settings
GRAD_ACCUMULATION=4

mkdir -p $FEATURE_DIR
mkdir -p $SAVE_DIR

# =================================================================
# STEP 6: Fine-tune CNN (Survival)
# =================================================================
echo "----------------------------------------------------------------"
echo "STEP 6: Fine-tuning CNN..."
echo "----------------------------------------------------------------"
python seattn_train/step6_train_survival_cnn.py \
    --csv_path $CSV_PATH \
    --patch_dir $PATCH_DIR \
    --image_dir $IMAGE_DIR \
    --phase1_model $PATHOLOGY_WEIGHTS \
    --save_dir $SAVE_DIR \
    --gc $GRAD_ACCUMULATION

# =================================================================
# STEP 4: Extract Features (with Survival-Tuned CNN)
# =================================================================
echo "----------------------------------------------------------------"
echo "STEP 4: Extracting Features..."
echo "----------------------------------------------------------------"
python seattn_train/step4_extract_features.py \
    --csv_path $CSV_PATH \
    --patch_dir $PATCH_DIR \
    --image_folder $IMAGE_DIR \
    --model_path "$SAVE_DIR/phase4_cnn.pth" \
    --feature_dir $FEATURE_DIR

# =================================================================
# STEP 7: Train Attention (Discrete Only)
# =================================================================
echo "----------------------------------------------------------------"
echo "STEP 7: Training Attention (Discrete Survival)..."
echo "----------------------------------------------------------------"
python seattn_train/step7_train_survival_attn.py \
    --train_csv $TRAIN_CSV \
    --val_csv $VAL_CSV \
    --feature_dir $FEATURE_DIR \
    --save_dir $SAVE_DIR \
    --epochs 50 \
    --gc $GRAD_ACCUMULATION

# =================================================================
# STEP 8: Fine-tune MLP (Cox) - Optional but Recommended
# =================================================================
echo "----------------------------------------------------------------"
echo "STEP 8: Fine-tuning MLP with Cox Loss..."
echo "----------------------------------------------------------------"
python seattn_train/step8_train_survival_mlp.py \
    --train_csv $TRAIN_CSV \
    --val_csv $VAL_CSV \
    --feature_dir $FEATURE_DIR \
    --phase6_model "$SAVE_DIR/phase6_discrete.pth" \
    --save_dir $SAVE_DIR

echo "----------------------------------------------------------------"
echo "Survival Pipeline Complete!"
echo "----------------------------------------------------------------"