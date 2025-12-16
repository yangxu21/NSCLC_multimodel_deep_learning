#!/bin/bash

# -----------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------

# 1. Data Paths (CSVs must be pre-merged with all columns)
TRAIN_CSV="data/mmdl_train.csv"
VAL_CSV="data/mmdl_val.csv"

# 2. Feature Directories
# NOTE: These must be 512-dim patient-level vectors (1 per slide/patient)
# stored as .pt files (e.g., 'SLIDE_123.pt')
WSI_FEATS_512="processed_data/wsi_patient_vectors_512"

# 3. Output Directories
SAVE_DIR="checkpoints/mmdl"
mkdir -p $SAVE_DIR

# -----------------------------------------------------------------
# STEP 1: Train Individual Modality Models (Cox)
# -----------------------------------------------------------------
echo "----------------------------------------------------------------"
echo "STEP 1: Training Individual Cox Models for 7 Modalities..."
echo "----------------------------------------------------------------"

# List of modalities to train individually
# These names are used for saving checkpoints (e.g., mutation_data_best.pth)
MODALITIES=(
    "mutation_data"
    "mutation_gene_set"
    "cna_amp_data"
    "cna_del_data"
    "amp_gene_set"
    "del_gene_set"
    "clinical_data"
)

for MOD in "${MODALITIES[@]}"; do
    echo ">> Training Individual Model: $MOD"
    
    python ngs_train/step1_train_individual.py \
        --modality_name "$MOD" \
        --train_csv $TRAIN_CSV \
        --val_csv $VAL_CSV \
        --save_dir "$SAVE_DIR/individual" \
        --batch_size 64
done

# -----------------------------------------------------------------
# STEP 2: Train MMDL Fusion Model (WSI + NGS + Clinical)
# -----------------------------------------------------------------
echo "----------------------------------------------------------------"
echo "STEP 2: Training Multi-Modal Fusion Model..."
echo "----------------------------------------------------------------"

python ngs_train/step2_train_fusion_mmdl.py \
    --train_csv $TRAIN_CSV \
    --val_csv $VAL_CSV \
    --wsi_feature_dir $WSI_FEATS_512 \
    --save_dir "$SAVE_DIR/fusion"

echo "----------------------------------------------------------------"
echo "Multimodal Pipeline Completed Successfully!"
echo "Checkpoints saved in $SAVE_DIR"
echo "----------------------------------------------------------------"