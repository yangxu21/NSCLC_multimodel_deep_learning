"""
Step 3: Train CNN on Pathology (lung cancer subtype) Labels
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from models.seattn import ResNet_MIL

import warnings
Image.MAX_IMAGE_PIXELS = None 
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

class WSI_Dataset(Dataset):
    def __init__(self, df, patch_folder, image_folder, transform=None):
        self.df = df
        self.patch_folder = patch_folder
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = str(row['slide_id'])
        h5_path = os.path.join(self.patch_folder, slide_id + '.h5')
        
        try:
            bag = []
            with h5py.File(h5_path, 'r') as f:
                if 'imgs' in f:
                    n = f['imgs'].shape[0]
                    indices = np.sort(np.random.choice(n, 192, replace=(n < 192)))
                    imgs = f['imgs'][indices]
                    bag = [self.transform(Image.fromarray(img)) for img in imgs]
                elif 'coords' in f:
                    coords = f['coords'][:]
                    n = len(coords)
                    indices = np.random.choice(n, 192, replace=(n < 192))
                    selected_coords = coords[indices]
                    img_path = os.path.join(self.image_folder, slide_id + '.jpg')
                    with Image.open(img_path).convert('RGB') as wsi_image:
                        for (x, y) in selected_coords:
                            # Use native patch size (e.g., 256)
                            patch = wsi_image.crop((x, y, x+256, y+256))
                            bag.append(self.transform(patch))
                else:
                    return None
            return torch.stack(bag), torch.tensor(int(row['label'])).float(), torch.tensor(n)
        except Exception as e:
            return None

def train(args):
    df = pd.read_csv(args.csv_path)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.76271061, 0.62484627, 0.7082412],
                             std=[0.16601817, 0.19249444, 0.15693703])
    ])
    
    loader = DataLoader(
        WSI_Dataset(df, args.patch_folder, args.image_folder, transform), 
        batch_size=1, shuffle=True, num_workers=4, 
        collate_fn=lambda x: torch.utils.data.dataloader.default_collate(list(filter(None, x))) if list(filter(None, x)) else None
    )
    
    model = ResNet_MIL(n_classes=1).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(30):
        model.train()
        eloss = 0.
        count = 0
        threshold = 192 * (min(epoch, 29) * 2 + 1)
        
        for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            if batch is None: continue
            data, label, n = batch
            
            if n.item() < threshold: continue
            
            loss = criterion(model(data.cuda()).view(-1), label.cuda().view(-1)) / args.gc
            loss.backward()
            eloss += loss.item() * args.gc
            count += 1
            
            if (i+1) % args.gc == 0:
                optimizer.step()
                optimizer.zero_grad()
                
        avg_loss = eloss / max(count, 1)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'phase1_cnn.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--patch_folder', required=True)
    parser.add_argument('--image_folder', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--gc', type=int, default=4, help='Gradient accumulation steps')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)