"""
Step 7: Train Attention-based Discrete-Time Survival Model (discrete labels)
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.survival import attention_discrete_time_survival_model
from utility.losses import NLLSurvLoss
from utility.utils import process_time_data
from utility.early_stopping import EarlyStopping

class SurvFeatDS(Dataset):
    def __init__(self, df, fdir):
        self.df = df
        self.fdir = fdir
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = os.path.join(self.fdir, str(row['slide_id'])+'.pt')
        try:
            feats = torch.load(p)
        except:
            # Fallback for missing files during testing
            feats = torch.zeros(10, 2048) 
            
        l = int(row['time_interval_label'])
        c = int(row['censor'])
        return feats, l, c

def train(args):
    # Process Data for Discrete Labels
    df_tr = process_time_data(pd.read_csv(args.train_csv))
    df_val = process_time_data(pd.read_csv(args.val_csv))
    
    model = attention_discrete_time_survival_model(n_classes=4, n_features=[256, 32]).cuda()
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    criterion = NLLSurvLoss()
    stopper = EarlyStopping(patience=10, path=os.path.join(args.save_dir, 'phase6_discrete.pth'))

    BS = 1
    
    for epoch in range(args.epochs):
        model.train()
        eloss = 0.
        count = 0
        
        # Training Loop
        for i, batch in enumerate(DataLoader(SurvFeatDS(df_tr, args.feature_dir), batch_size=BS, shuffle=True)):
            d, l, c = batch
            d = d.squeeze(0).cuda()
            l, c = l.cuda(), c.cuda()
            
            h, _, _ = model(d)
            
            # Gradient Accumulation
            loss = criterion(h, l, c) / args.gc
            loss.backward()
            eloss += loss.item() * args.gc
            count += 1
            
            if (i + 1) % args.gc == 0:
                optimizer.step()
                optimizer.zero_grad()
            
        # Validation Loop
        model.eval()
        vloss = 0.
        vcount = 0
        with torch.no_grad():
            for d, l, c in DataLoader(SurvFeatDS(df_val, args.feature_dir), batch_size=BS):
                d = d.squeeze(0).cuda()
                l, c = l.cuda(), c.cuda()
                h, _, _ = model(d)
                vloss += criterion(h, l, c).item()
                vcount += 1
        
        avg_val = vloss / max(vcount, 1)
        print(f"Ep {epoch+1} Val Loss: {avg_val:.4f}")
        stopper(avg_val, model)
        if stopper.early_stop: break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--feature_dir', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--gc', type=int, default=1, help='Gradient accumulation steps')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)