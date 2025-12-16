"""
Step 4: Extract Features using Trained CNN
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import h5py
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from models.seattn import ResNet_MIL

Image.MAX_IMAGE_PIXELS = None 

class InferenceDataset(Dataset):
    def __init__(self, h5_path, image_path, transform):
        self.h5_path = h5_path
        self.image_path = image_path
        self.transform = transform
        self.use_coords = False
        
        with h5py.File(h5_path, 'r') as f:
            if 'imgs' in f:
                self.len = f['imgs'].shape[0]
            elif 'coords' in f:
                self.len = f['coords'].shape[0]
                self.use_coords = True
                self.coords = f['coords'][:] 
            else:
                self.len = 0
                
    def __len__(self):
        return self.len
        
    def __getitem__(self, idx):
        if self.use_coords:
            with Image.open(self.image_path) as wsi_image:
                wsi_image = wsi_image.convert('RGB')
                x, y = self.coords[idx]
                # Crop at native size (256)
                img = wsi_image.crop((x, y, x+256, y+256))
        else:
            with h5py.File(self.h5_path, 'r') as f:
                img_arr = f['imgs'][idx]
            img = Image.fromarray(img_arr)
            
        return self.transform(img)

def extract(args):
    df = pd.read_csv(args.csv_path)
    model = ResNet_MIL().cuda()
    model.load_state_dict(torch.load(args.model_path), strict=False)
    model.feature_extractor.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.76271061, 0.62484627, 0.7082412],
                             std=[0.16601817, 0.19249444, 0.15693703])
    ])
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sid = str(row['slide_id'])
        h5 = os.path.join(args.patch_dir, sid + '.h5')
        img_path = os.path.join(args.image_folder, sid + '.jpg') 
        
        if not os.path.exists(h5): continue
            
        try:
            ds = InferenceDataset(h5, img_path, transform)
            if len(ds) == 0: 
                continue
            
            dl = DataLoader(ds, batch_size=128, num_workers=4, shuffle=False)
            feats = []
            with torch.no_grad():
                for imgs in dl:
                    f = model.feature_extractor(imgs.cuda())
                    feats.append(f.view(f.size(0), -1).cpu().numpy())
            
            if feats:
                torch.save(torch.from_numpy(np.concatenate(feats)), os.path.join(args.feature_dir, sid + '.pt'))
        except Exception as e:
            print(f"Error {sid}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--patch_dir', required=True)
    parser.add_argument('--image_folder', required=True) 
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--feature_dir', required=True)
    args = parser.parse_args()
    os.makedirs(args.feature_dir, exist_ok=True)
    extract(args)