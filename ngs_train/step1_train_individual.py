"""
Step 1: Train Individual Cox Models for Each Modality
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.multimodal import IndividualCoxModel
from utility.early_stopping import EarlyStopping
from torchsurv.loss.cox import neg_partial_log_likelihood

# Format: 'modality_name': (start_index, end_index)
MODALITY_INDICES = {
    'mutation_data':     (4, 104),   # 100 features
    'mutation_gene_set': (104, 204), # 100 features
    'cna_amp_data':      (204, 304), # 100 features
    'cna_del_data':      (304, 404), # 100 features
    'amp_gene_set':      (404, 504), # 100 features
    'del_gene_set':      (504, 604), # 100 features
    'clinical_data':     (604, 609)  # 5 features
}

class NGSDataset(Dataset):
    def __init__(self, csv_path, modality_name):
        self.df = pd.read_csv(csv_path)
        
        # Determine column slice based on modality name
        if modality_name not in MODALITY_INDICES:
            raise ValueError(f"Unknown modality: {modality_name}. Available: {list(MODALITY_INDICES.keys())}")
            
        start, end = MODALITY_INDICES[modality_name]
        
        # Slice features
        self.features = self.df.iloc[:, start:end].values.astype(float)
        
        # Targets
        self.events = self.df['dead'].values
        self.times = self.df['survival_months'].values
        
    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        
        # Cox Target: Event and Time
        c_raw = int(self.events[idx])
        e = 1.0 if c_raw == 1 else 0.0
        t = float(self.times[idx])
        
        return x, torch.tensor(e).float(), torch.tensor(t).float()

def train(args):
    # 1. Setup DataLoaders
    print(f"Loading data for modality: {args.modality_name}...")
    train_ds = NGSDataset(args.train_csv, args.modality_name)
    val_ds = NGSDataset(args.val_csv, args.modality_name)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # 2. Model Setup
    input_dim = train_ds.features.shape[1]
    print(f"Input Dimension: {input_dim}")
    
    # Use SNN for CNA data, MLP for others
    block_type = 'SNN' if 'cna' in args.modality_name.lower() else 'MLP'
    
    # Smaller network for Clinical data (only 5 features)
    sizes = [64, 32] if 'clinical' in args.modality_name.lower() else [256, 32]
    
    model = IndividualCoxModel(input_dim=input_dim, sizes=sizes, block_type=block_type).cuda()
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 833], gamma=0.5)
    
    # Save best model with modality name
    save_path = os.path.join(args.save_dir, f'{args.modality_name}_best.pth')
    stop = EarlyStopping(patience=50, path=save_path)

    # 3. Training Loop
    print("Starting Training...")
    for epoch in range(1000): 
        model.train()
        train_loss = 0.
        
        for x, e, t in train_loader:
            x, e, t = x.cuda(), e.cuda(), t.cuda()
            
            risk = model(x).squeeze()
            
            # Cox Loss
            loss = neg_partial_log_likelihood(risk, e.bool(), t)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        if scheduler: scheduler.step()

        # Validation
        model.eval()
        vloss = 0.
        with torch.no_grad():
            for x, e, t in val_loader:
                x, e, t = x.cuda(), e.cuda(), t.cuda()
                risk = model(x).squeeze()
                vloss += neg_partial_log_likelihood(risk, e.bool(), t).item()
        
        avg_val = vloss / len(val_loader)
        avg_trn = train_loss / len(train_loader)
        
        if epoch % 100 == 0:
            print(f"Ep {epoch:04d} | Train: {avg_trn:.4f} | Val: {avg_val:.4f}")
        
        stop(avg_val, model)
        if stop.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality_name', required=True, help="Name must match keys in MODALITY_INDICES")
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)