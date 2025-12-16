"""
Step 2: Patch Extraction from Resized Images
"""

import sys
import os
import argparse
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wsi_core.batch_process_utils import initialize_df
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchPatches

def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir, 
                  patch_size=256, step_size=256, 
                  seg_params=None, filter_params=None, vis_params=None, patch_params=None,
                  auto_skip=True):
    
    # Process image files generated in Step 1
    slides = sorted([f for f in os.listdir(source) if f.endswith('.jpg') or f.endswith('.png')])
    
    # Initialize DataFrame
    if seg_params is None: seg_params = {}
    if vis_params is None: vis_params = {}

    seg_params['seg_level'] = -1
    vis_params['vis_level'] = -1
    
    df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
    
    mask = df['process'] == 1
    process_stack = df[mask]
    total = len(process_stack)

    print(f"Found {total} resized slides to process.")

    for i in range(total):
        idx = process_stack.index[i]
        slide = process_stack.loc[idx, 'slide_id']
        slide_id = os.path.splitext(slide)[0]
        
        print(f"\n[{i+1}/{total}] Processing {slide_id}")
        
        # Check auto-skip
        h5_path = os.path.join(patch_save_dir, slide_id + '.h5')
        if auto_skip and os.path.isfile(h5_path):
            print('  - Patches already extracted. Skipping...')
            continue

        full_path = os.path.join(source, slide)
        
        try:
            # Load the Resized Image
            WSI_object = WholeSlideImage(full_path)
            
            # 1. Segment Tissue
            WSI_object.segmentTissue(**seg_params, filter_params=filter_params)
            
            # 2. Save Mask Visualization
            mask_img = WSI_object.visWSI(**vis_params)
            mask_path = os.path.join(mask_save_dir, slide_id+'.jpg')
            mask_img.save(mask_path)
            print(f"  - Segmentation mask saved to {mask_path}")

            # 3. Extract Patches
            patch_params.update({'patch_size': patch_size, 'step_size': step_size, 'save_path': patch_save_dir})
            file_path = WSI_object.createPatches_bag_hdf5(**patch_params, save_coord=True)
            print(f"  - Patch coordinates saved to {file_path}")
            
            # 4. Stitching (Verification)
            if os.path.isfile(file_path):
                # Downscale for visualization
                heatmap = StitchPatches(file_path, downscale=16, bg_color=(0,0,0), alpha=-1, draw_grid=False)
                stitch_path = os.path.join(stitch_save_dir, slide_id+'.jpg')
                heatmap.save(stitch_path)
                print(f"  - Reconstructed stitch saved to {stitch_path}")

        except Exception as e:
            print(f"  [Error] Failed processing {slide_id}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Step 2: Patch Extraction from Resized Images')
    parser.add_argument('--source', type=str, required=True, help='Path to folder containing Step 1 outputs (JPG/PNG)')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to output directory')
    
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--step_size', type=int, default=256)
    parser.add_argument('--preset', type=str, default='bwh_biopsy', choices=['bwh_biopsy', 'tcga'])

    args = parser.parse_args()

    # Define Parameters
    if args.preset == 'bwh_biopsy':
        seg_params = {'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False, 'keep_ids': 'none', 'exclude_ids': 'none'}
        filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 8}
    elif args.preset == 'tcga':
        seg_params = {'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': True, 'keep_ids': 'none', 'exclude_ids': 'none'}
        filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 8}

    # Setup directories
    patch_save_dir = os.path.join(args.save_dir, 'patches')
    mask_save_dir = os.path.join(args.save_dir, 'masks')
    stitch_save_dir = os.path.join(args.save_dir, 'stitches')

    os.makedirs(patch_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)
    os.makedirs(stitch_save_dir, exist_ok=True)

    seg_and_patch(
        source=args.source, 
        save_dir=args.save_dir, 
        patch_save_dir=patch_save_dir, 
        mask_save_dir=mask_save_dir, 
        stitch_save_dir=stitch_save_dir, 
        patch_size=args.patch_size, 
        step_size=args.step_size, 
        seg_params=seg_params,
        filter_params=filter_params,
        vis_params={'line_thickness': 250},
        patch_params={'use_padding': True, 'contour_fn': 'four_pt'}
    )