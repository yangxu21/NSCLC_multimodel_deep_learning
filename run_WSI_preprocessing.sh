#!/bin/bash

# =================================================================
# CONFIGURATION
# =================================================================

# Input/Output Directories
RAW_WSI_DIR="data/raw_wsi"                 
RESIZED_DIR="processed_data/resized"
PATCHES_DIR="processed_data/patches_output"        

# Parameters
PATCH_SIZE=256
STEP_SIZE=256
TARGET_SIZE=0.55

# Create output directories if they don't exist
mkdir -p $RESIZED_DIR
mkdir -p $PATCHES_DIR

# =================================================================
# STEP 1: Resize WSI
# =================================================================
echo "----------------------------------------------------------------"
echo "STEP 1: Resizing Whole Slide Images..."
echo "----------------------------------------------------------------"

python preprocessing_WSI/step1_resize_wsi.py \
    --source $RAW_WSI_DIR \
    --save_dir $RESIZED_DIR \
    --target_um $TARGET_SIZE

# =================================================================
# STEP 2: Create Patches
# =================================================================
echo "----------------------------------------------------------------"
echo "STEP 2: Creating Patches from Resized Images..."
echo "----------------------------------------------------------------"

python preprocessing_WSI/step2_create_patches.py \
    --source $RESIZED_DIR \
    --save_dir $PATCHES_DIR \
    --patch_size $PATCH_SIZE \
    --step_size $STEP_SIZE

echo "================================================================"
echo "Preprocessing Complete!"
echo "Resized images saved to: $RESIZED_DIR"
echo "Patches saved to:        $PATCHES_DIR"
echo "================================================================"