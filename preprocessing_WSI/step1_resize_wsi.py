"""
Step 1: Resolution-Based WSI Resizing
-------------------------------------
Resizes Whole Slide Images (SVS/TIF) to a specific target resolution (default 0.55 um/px).
Uses a 4x4 grid chunking strategy to manage memory usage during resizing.
"""

import os
import argparse
import numpy as np
import cv2
import openslide
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None # Allow processing of very large images

def resize_wsi(image_path, output_folder, target_um=0.55, output_format='jpg'):
    filename = os.path.basename(image_path)
    image_name = os.path.splitext(filename)[0]
    output_path = os.path.join(output_folder, f"{image_name}.{output_format}")

    if os.path.exists(output_path): return

    try:
        slide = openslide.OpenSlide(image_path)
    except Exception as e:
        print(f"Error opening {filename}: {e}")
        return

    # 1. Calculate Scale based on MPP (Microns Per Pixel)
    try:
        mpp = float(slide.properties['openslide.mpp-x'])
        scale = mpp / target_um
    except KeyError:
        # Fallback if MPP is missing (assume 40x = 0.25 mpp)
        print(f"Warning: MPP missing for {filename}, assuming 0.25")
        scale = 0.25 / target_um

    width, height = slide.dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)

    # 2. Create Blank Canvas
    new_image = np.zeros((new_height + 4, new_width + 4, 3), dtype=np.uint8)
    patches = []

    # 3. Process in 4x4 Grid
    chunk_w, chunk_h = width // 4, height // 4
    new_chunk_w, new_chunk_h = new_width // 4, new_height // 4

    for i in range(4):
        for j in range(4):
            patch = slide.read_region((i * chunk_w, j * chunk_h), 0, (chunk_w, chunk_h))
            patch_arr = np.array(patch)[:, :, :3]
            patch_resized = cv2.resize(patch_arr, (new_chunk_w, new_chunk_h), interpolation=cv2.INTER_CUBIC)
            patches.append(patch_resized)
    slide.close()

    # 4. Stitch Back Together
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            y_s, y_e = j * new_chunk_h, (j + 1) * new_chunk_h
            x_s, x_e = i * new_chunk_w, (i + 1) * new_chunk_w
            new_image[y_s:y_e, x_s:x_e] = patches[idx]

    # 5. Save
    try:
        Image.fromarray(new_image).save(output_path, quality=90, optimize=True)
    except Exception as e:
        print(f"Save failed: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--target_um', type=float, default=0.55)
    parser.add_argument('--format', default='jpg')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    files = sorted([os.path.join(args.source, f) for f in os.listdir(args.source) if f.endswith(('.svs', '.sv', '.tif', '.ndpi'))])
    for f in tqdm(files): resize_wsi(f, args.save_dir, args.target_um, args.format)