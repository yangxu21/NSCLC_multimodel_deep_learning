import os
import cv2
import numpy as np
from PIL import Image
import h5py

from .wsi_utils import isBlackPatch, isWhitePatch
from .util_classes import isInContourV1, isInContourV2, isInContourV3_Easy, isInContourV3_Hard, Contour_Checking_fn
from .file_utils import load_pkl, save_pkl

# Disable limit for large resized JPGs
Image.MAX_IMAGE_PIXELS = None

class WholeSlideImage(object):
    def __init__(self, path):
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.wsi = Image.open(path).convert("RGB")
        self.level_dim = self.wsi.size
        self.contours_tissue = None
        self.holes_tissue = None

    def segmentTissue(self, sthresh=20, sthresh_up=255, mthresh=7, close=0, use_otsu=False, 
                      filter_params={'a_t':100}, ref_patch_size=512, exclude_ids=[], keep_ids=[], **kwargs):
        
        def _filter_contours(contours, hierarchy, filter_params):
            filtered = []
            hierarchy_1 = np.flatnonzero(hierarchy[:,1] == -1)
            all_holes = []
            for cont_idx in hierarchy_1:
                cont = contours[cont_idx]
                holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
                a = cv2.contourArea(cont)
                hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
                a = a - np.array(hole_areas).sum()
                if a == 0: continue
                if tuple((filter_params['a_t'],)) < tuple((a,)): 
                    filtered.append(cont_idx)
                    all_holes.append(holes)
            foreground_contours = [contours[cont_idx] for cont_idx in filtered]
            hole_contours = []
            for hole_ids in all_holes:
                unfiltered_holes = [contours[idx] for idx in hole_ids ]
                unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
                unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
                filtered_holes = []
                for hole in unfilered_holes:
                    if cv2.contourArea(hole) > filter_params['a_h']:
                        filtered_holes.append(hole)
                hole_contours.append(filtered_holes)
            return foreground_contours, hole_contours

        img = np.array(self.wsi)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_med = cv2.medianBlur(img_hsv[:,:,1], mthresh)
        
        if use_otsu:
            _, img_otsu = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        else:
            _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)

        if close > 0:
            kernel = np.ones((close, close), np.uint8)
            img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)                 

        contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        
        if hierarchy is None:
            self.contours_tissue = []
            self.holes_tissue = []
            return

        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
        if filter_params: 
            foreground_contours, hole_contours = _filter_contours(contours, hierarchy, filter_params)
        else:
            foreground_contours, hole_contours = contours, []

        self.contours_tissue = foreground_contours
        self.holes_tissue = hole_contours

    def visWSI(self, color=(0,255,0), hole_color=(0,0,255), annot_color=(255,0,0), 
               line_thickness=250, seg_display=True, **kwargs):
        img = np.array(self.wsi)
        thickness = max(2, int(line_thickness / 20))
        if self.contours_tissue is not None and seg_display:
            cv2.drawContours(img, self.contours_tissue, -1, color, thickness, lineType=cv2.LINE_8)
            for holes in self.holes_tissue:
                cv2.drawContours(img, holes, -1, hole_color, thickness, lineType=cv2.LINE_8)
        return Image.fromarray(img)

    def createPatches_bag_hdf5(self, save_path, patch_size=256, step_size=256, save_coord=True, **kwargs):
        # 1. Define the output file path string
        file_path = os.path.join(save_path, self.name + '.h5')
        
        # 2. Collect all coordinates first
        coords_list = []
        contours = self.contours_tissue
        
        for idx, cont in enumerate(contours):
            patch_gen = self._getPatchGenerator(cont, idx, save_path, patch_size, step_size, **kwargs)
            for patch in patch_gen:
                coords_list.append((patch['x'], patch['y']))

        # 3. Save to HDF5
        with h5py.File(file_path, "w") as f:
            dset = f.create_dataset('coords', (len(coords_list), 2), dtype='int32')
            if len(coords_list) > 0:
                dset[:] = coords_list
            
            # Save metadata attributes
            dset.attrs['patch_size'] = patch_size
            dset.attrs['patch_level'] = 0
            dset.attrs['downsample'] = (1.0, 1.0)
            dset.attrs['level_dim'] = self.wsi.size
            dset.attrs['wsi_dimensions'] = self.wsi.size
            dset.attrs['name'] = self.name
            dset.attrs['save_path'] = save_path

        # 4. RETURN THE PATH STRING
        return file_path

    def _getPatchGenerator(self, cont, cont_idx, save_path, patch_size=256, step_size=256, 
        white_black=True, white_thresh=15, black_thresh=50, contour_fn='four_pt', use_padding=True, **kwargs):
        
        img_w, img_h = self.wsi.size
        start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (0, 0, img_w, img_h)
        
        if isinstance(contour_fn, str):
            if contour_fn == 'four_pt':
                cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=patch_size, center_shift=0.5)
            elif contour_fn == 'center':
                cont_check_fn = isInContourV2(contour=cont, patch_size=patch_size)
            else:
                raise NotImplementedError
        else:
            cont_check_fn = contour_fn

        if use_padding:
            stop_y = start_y + h
            stop_x = start_x + w
        else:
            stop_y = min(start_y + h, img_h - patch_size)
            stop_x = min(start_x + w, img_w - patch_size)

        for y in range(start_y, stop_y, step_size):
            for x in range(start_x, stop_x, step_size):
                if not self.isInContours(cont_check_fn, (x,y), self.holes_tissue[cont_idx], patch_size): 
                    continue    
                
                if white_black:
                    patch_PIL = self.wsi.crop((x, y, x+patch_size, y+patch_size))
                    if isBlackPatch(np.array(patch_PIL), rgbThresh=black_thresh) or isWhitePatch(np.array(patch_PIL), satThresh=white_thresh): 
                        continue
                
                patch_info = {
                    'x': x, 'y': y, 
                    'cont_idx': cont_idx, 
                    'name': self.name, 
                    'save_path': save_path
                }
                yield patch_info

    @staticmethod
    def isInHoles(holes, pt, patch_size):
        for hole in holes:
            if cv2.pointPolygonTest(hole, (pt[0]+patch_size/2, pt[1]+patch_size/2), False) > 0: return 1
        return 0

    @staticmethod
    def isInContours(cont_check_fn, pt, holes=None, patch_size=256):
        if cont_check_fn(pt):
            if holes is not None:
                return not WholeSlideImage.isInHoles(holes, pt, patch_size)
            else: return 1
        return 0