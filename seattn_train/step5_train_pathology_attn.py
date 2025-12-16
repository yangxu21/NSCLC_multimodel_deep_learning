"""
Step 5: Train Attention-based MIL on Extracted Pathology Features
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models.seattn import MIL_attn
from utility.early_stopping import EarlyStopping

class FeatDS(Dataset):
    def __init__(self, df, fdir):
        self.df = df
        self.fdir = fdir
    def __len__(self): 
        return len(self.df)
    def __getitem__(self, idx):
        p = os.path.join(self.fdir, str(self.df.iloc[idx]['slide_id']) + '.pt')
        try:
            return torch.load(p), torch.tensor(int(self.df.iloc[idx]['label'])).float()
        except:
            return torch.zeros(10, 2048), torch.tensor(0).float()

def train(args):
    df_tr = pd.read_csv(args.train_csv)
    df_val = pd.read_csv(args.val_csv)
    
    model = MIL_attn(n_classes=1, dropout=0.25).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    stopper = EarlyStopping(patience=10, path=os.path.join(args.save_dir, 'phase3_attn.pth'))
    
    for epoch in range(args.epochs):
        model.train()
        for i, (d, l) in enumerate(DataLoader(FeatDS(df_tr, args.feature_dir), batch_size=1, shuffle=True)):
            limit = int(1024 * (1.5 ** ((epoch % 10) + 1)))
            d = d.squeeze(0)
            if d.size(0) > limit: 
                d = d[torch.randperm(d.size(0))[:limit]]

            loss = criterion(model(d.cuda()).view(-1), l.cuda().view(-1)) / args.gc
            loss.backward()
            
            if (i + 1) % args.gc == 0:
                optimizer.step()
                optimizer.zero_grad()
            
        model.eval()
        vloss = 0.
        count = 0
        with torch.no_grad():
            for d, l in DataLoader(FeatDS(df_val, args.feature_dir), batch_size=1):
                vloss += criterion(model(d.cuda().squeeze(0)).view(-1), l.cuda().view(-1)).item()
                count += 1
        
        avg_loss = vloss / max(count, 1)
        print(f"Ep {epoch+1} Val: {avg_loss:.4f}")
        stopper(avg_loss, model)
        if stopper.early_stop: break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--feature_dir', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--gc', type=int, default=4, help='Gradient accumulation steps')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)