import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.multimodal import MMDL_Fusion
from utility.early_stopping import EarlyStopping
from torchsurv.loss.cox import neg_partial_log_likelihood


# Dataset Class
class FusionDataset(Dataset):
    """
    Loads aligned Multi-Modal data:
    1. Tabular data (NGS + Clinical) from CSV.
    2. Pre-extracted WSI vectors (512-dim) from .pt files.
    """
    def __init__(self, csv_path, wsi_feature_dir):
        self.df = pd.read_csv(csv_path)
        self.wsi_dir = wsi_feature_dir
        self.mut_data = self.df.iloc[:, 4:104].values.astype(float)    
        self.mut_set  = self.df.iloc[:, 104:204].values.astype(float)  
        self.cna_amp  = self.df.iloc[:, 204:304].values.astype(float)  
        self.cna_del  = self.df.iloc[:, 304:404].values.astype(float)  
        self.amp_set  = self.df.iloc[:, 404:504].values.astype(float)  
        self.del_set  = self.df.iloc[:, 504:604].values.astype(float)  
        self.clinical = self.df.iloc[:, 604:609].values.astype(float)
        
        self.slide_ids = self.df['slide_id'].values
        self.events = self.df['dead'].values
        self.times = self.df['survival_months'].values

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        sid = str(self.slide_ids[idx])
        
        # --- Load Pre-extracted WSI Feature (512-dim) ---
        pt_path = os.path.join(self.wsi_dir, sid + '.pt')
        try:
            # Load and ensure it is a 1D vector of size 512
            wsi = torch.load(pt_path)
            if wsi.dim() > 1:
                wsi = wsi.view(-1)
        except Exception as e:
            wsi = torch.zeros(512)

        # --- Process Targets ---
        event = 1.0 if self.events[idx] == 1 else 0.0
        time = float(self.times[idx])

        return (
            wsi.float(),
            torch.tensor(self.cna_amp[idx]).float(),
            torch.tensor(self.cna_del[idx]).float(),
            torch.tensor(self.mut_data[idx]).float(),
            torch.tensor(self.amp_set[idx]).float(),
            torch.tensor(self.del_set[idx]).float(),
            torch.tensor(self.mut_set[idx]).float(),
            torch.tensor(self.clinical[idx]).float(),
            torch.tensor(event).float(),
            torch.tensor(time).float()
        )

# Training Loop
def train(args):
    # 1. Setup DataLoaders
    train_ds = FusionDataset(args.train_csv, args.wsi_feature_dir)
    val_ds = FusionDataset(args.val_csv, args.wsi_feature_dir)
    
    loader_args = {'batch_size': 32, 'shuffle': True, 'num_workers': 4, 'drop_last': True}
    val_loader_args = {'batch_size': 32, 'shuffle': False, 'num_workers': 4}
    
    train_loader = DataLoader(train_ds, **loader_args)
    val_loader = DataLoader(val_ds, **val_loader_args)
    
    # 2. Initialize Model
    model = MMDL_Fusion(block_type='MLP').cuda()
    
    # 3. Optimization
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 83], gamma=0.5)
    stop = EarlyStopping(patience=20, path=os.path.join(args.save_dir, 'mmdl_fusion_best.pth'))

    print(f"Starting MMDL Training on {len(train_ds)} samples...")
    
    for epoch in range(100): 
        model.train()
        train_loss = 0.
        
        for batch in train_loader:
            wsi, ca, cd, md, as_, ds, ms, clin, e, t = [x.cuda() for x in batch]
            
            # Forward Pass
            # Output shape: (Batch_Size, 1) -> Squeeze to (Batch_Size)
            risk = model(wsi, ca, cd, md, as_, ds, ms, clin).squeeze()
            
            # Cox Loss (Risk, Event=Bool, Time)
            loss = neg_partial_log_likelihood(risk, e.bool(), t)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        if scheduler: scheduler.step()

        # Validation Loop
        model.eval()
        vloss = 0.
        with torch.no_grad():
            for batch in val_loader:
                wsi, ca, cd, md, as_, ds, ms, clin, e, t = [x.cuda() for x in batch]
                
                risk = model(wsi, ca, cd, md, as_, ds, ms, clin).squeeze()
                vloss += neg_partial_log_likelihood(risk, e.bool(), t).item()
        
        # Logging & Early Stopping
        avg_trn = train_loss / len(train_loader)
        avg_val = vloss / len(val_loader)
        
        print(f"Ep {epoch:03d} | Train: {avg_trn:.4f} | Val: {avg_val:.4f}")
        
        stop(avg_val, model)
        if stop.early_stop:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True, help="Path to training CSV with all modalities")
    parser.add_argument('--val_csv', required=True, help="Path to validation CSV")
    parser.add_argument('--wsi_feature_dir', required=True, help="Directory containing .pt files (512-dim)")
    parser.add_argument('--save_dir', required=True, help="Directory to save checkpoints")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)