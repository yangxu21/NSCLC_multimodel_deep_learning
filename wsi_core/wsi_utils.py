import h5py
import numpy as np
import os
import cv2
from .util_classes import Mosaic_Canvas

def isWhitePatch(patch, satThresh=5):
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    return True if np.mean(patch_hsv[:,:,1]) < satThresh else False

def isBlackPatch(patch, rgbThresh=40):
    return True if np.all(np.mean(patch, axis = (0,1)) < rgbThresh) else False

def isBlackPatch_S(patch, rgbThresh=20, percentage=0.05):
    num_pixels = patch.size[0] * patch.size[1]
    return True if np.all(np.array(patch) < rgbThresh, axis=(2)).sum() > num_pixels * percentage else False

def isWhitePatch_S(patch, rgbThresh=220, percentage=0.2):
    num_pixels = patch.size[0] * patch.size[1]
    return True if np.all(np.array(patch) > rgbThresh, axis=(2)).sum() > num_pixels * percentage else False

def coord_generator(x_start, x_end, x_step, y_start, y_end, y_step, args_dict=None):
    for x in range(x_start, x_end, x_step):
        for y in range(y_start, y_end, y_step):
            if args_dict is not None:
                process_dict = args_dict.copy()
                process_dict.update({'pt': (x,y)})
                yield process_dict
            else:
                yield (x,y)

def save_hdf5(output_path, asset_dict, attr_dict=None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path

def initialize_hdf5_bag(first_patch, save_coord=False):
    x = np.array([first_patch])
    int8_type = h5py.special_dtype(vlen=np.dtype('int8'))
    data_shape = x.shape
    chunk_shape = (1, ) + data_shape[1:]
    maxshape = (None, ) + data_shape[1:]
    ds_dict = {
        'imgs': h5py.special_dtype(vlen=np.dtype('uint8')),
        'coords': h5py.special_dtype(vlen=np.dtype('int32'))
    }
    return ds_dict

def savePatchIter_bag_hdf5(patch):
    x = np.array([patch])
    return x

def to_percentiles(scores):
    from scipy.stats import rankdata
    scores = rankdata(scores, 'average')/len(scores) * 100   
    return scores

def screen_coords(scores, coords, top_left, bot_right):
    bot_right = np.array(bot_right)
    top_left = np.array(top_left)
    mask = np.logical_and(np.all(coords >= top_left, axis=1), np.all(coords <= bot_right, axis=1))
    scores = scores[mask]
    coords = coords[mask]
    return scores, coords

def sample_indices(scores, k, start=0.48, end=0.52, convert_to_percentile=False, seed=1):
    np.random.seed(seed)
    if convert_to_percentile:
        end = np.percentile(scores, end*100)
        start = np.percentile(scores, start*100)
    
    potential_indices = np.where((scores >= start) & (scores <= end))[0]
    if len(potential_indices) == 0: 
        return np.array([])
    indices = np.random.choice(potential_indices, min(k, len(potential_indices)), replace=False)
    return indices

def StitchPatches(hdf5_file_path, downscale=16, draw_grid=False, bg_color=(255,255,255), alpha=-1):
    file = h5py.File(hdf5_file_path, 'r')
    dset = file['coords']
    coords = dset[:]
    w, h = dset.attrs['wsi_dimensions']
    
    print('original size: {} x {}'.format(w, h))
    w_resized = int(w / downscale)
    h_resized = int(h / downscale)
    print('resized size: {} x {}'.format(w_resized, h_resized))
    
    patch_size = dset.attrs['patch_size']
    patch_level = dset.attrs['patch_level']
    print('patch size: {}x{} patch level: {}'.format(patch_size, patch_size, patch_level))
    
    patch_size_resized = int(patch_size / downscale)
    print('resized patch size: {}x{}'.format(patch_size_resized, patch_size_resized))
    
    canvas = Mosaic_Canvas(patch_size=patch_size_resized, n=len(coords), downscale=1, n_per_row=int(w_resized/patch_size_resized), bg_color=bg_color, alpha=alpha)
    
    return canvas.canvas