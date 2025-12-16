"""
Step 6: Fine-tune CNN for Survival (Block 4 Only)
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import h5py
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from models.seattn import ResNet_MIL
from models.survival import discrete_time_to_event_model
from utility.losses import NLLSurvLoss
from utility.utils import process_time_data

import warnings
Image.MAX_IMAGE_PIXELS = None 
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

class CombinedModel(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.base.attn_model.classifier = torch.nn.Identity()
        self.head = discrete_time_to_event_model(n_classes=4) 

    def forward(self, x): 
        return self.head(self.base(x))

class SurvBagDS(Dataset):
    def __init__(self, df, pdir, idir, transform):
        self.df, self.pdir, self.idir, self.transform = df, pdir, idir, transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = str(row['slide_id'])
        h5 = os.path.join(self.pdir, sid + '.h5')
        try:
            bag = []
            with h5py.File(h5, 'r') as f:
                if 'imgs' in f:
                    n = f['imgs'].shape[0]
                    ix = np.sort(np.random.choice(n, 1024, replace=(n<1024)))
                    imgs = f['imgs'][ix]
                    bag = [self.transform(Image.fromarray(i)) for i in imgs]
                elif 'coords' in f:
                    coords = f['coords'][:]
                    n = len(coords)
                    ix = np.random.choice(n, 1024, replace=(n<1024))
                    sel_coords = coords[ix]
                    img_path = os.path.join(self.idir, sid+'.jpg')
                    with Image.open(img_path).convert('RGB') as wsi:
                        for x,y in sel_coords:
                            bag.append(self.transform(wsi.crop((x,y,x+256,y+256))))
                else: return None
            return torch.stack(bag), torch.tensor(int(row['time_interval_label'])), torch.tensor(int(row['censor'])), torch.tensor(n)
        except: return None

def train(args):
    df = process_time_data(pd.read_csv(args.csv_path))
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.76271061, 0.62484627, 0.7082412],
                             std=[0.16601817, 0.19249444, 0.15693703])
    ])
    
    loader = DataLoader(
        SurvBagDS(df, args.patch_dir, args.image_dir, transform), 
        batch_size=1, shuffle=True, num_workers=4, 
        collate_fn=lambda x: torch.utils.data.dataloader.default_collate(list(filter(None, x))) if list(filter(None, x)) else None
    )
    
    base = ResNet_MIL(n_classes=1, freeze_blocks_1_3=True).cuda()
    
    base.load_state_dict(torch.load(args.phase1_model), strict=False)
    
    model = CombinedModel(base).cuda()
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, weight_decay=5e-3)
    criterion = NLLSurvLoss()
    
    for epoch in range(30):
        model.train()
        eloss = 0.
        count = 0
        thresh = 1024 * (min(epoch, 7) * 2 + 1)
        
        for i, batch in enumerate(tqdm(loader)):
            if batch is None: 
                continue
            data, l, c, n = batch
            
            if n.item() < thresh: 
                continue
            
            h, _, _ = model(data.cuda())
            loss = criterion(h, l.cuda(), c.cuda()) / args.gc
            loss.backward()
            eloss += loss.item() * args.gc
            count += 1
            
            if (i+1) % args.gc == 0:
                optimizer.step()
                optimizer.zero_grad()
                
        print(f"Ep {epoch+1} Loss: {eloss/max(count, 1):.4f}")
        torch.save(model.base.state_dict(), os.path.join(args.save_dir, 'phase4_cnn.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--patch_dir', required=True)
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--phase1_model', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--gc', type=int, default=4, help='Gradient accumulation steps')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)