"""
Step 8: Fine-tune MLP Survival Model using Cox Loss
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import torch
import torch.optim as optim
from models.survival import attention_COX_model
from torchsurv.loss.cox import neg_partial_log_likelihood
from utility.early_stopping import EarlyStopping

def train(args):
    df_tr, df_val = pd.read_csv(args.train_csv), pd.read_csv(args.val_csv)
    
    # Initialize Cox Model
    model = attention_COX_model(n_classes=1, n_features=[256, 32]).cuda()
    
    # --- SAFE WEIGHT LOADING ---
    state_dict = torch.load(args.phase6_model)
    filtered_state_dict = {k: v for k, v in state_dict.items() if 'fc_4' not in k}
    msg = model.load_state_dict(filtered_state_dict, strict=False)
    print(f"Loaded Step 7 Weights. Missing keys: {msg.missing_keys}")
    # ---------------------------
    
    for p in model.atten_model.parameters(): p.requires_grad = False
    
    def get_reps(df, split_name="Data"):
        X, T, E = [], [], []
        model.eval()
        with torch.no_grad():
            for _, r in df.iterrows():
                p = os.path.join(args.feature_dir, str(r['slide_id'])+'.pt')
                try:
                    f = torch.load(p).cuda() 
                    x = model.atten_model.fc_1(f)
                    A, h = model.atten_model.attn_net(x)
                    A = torch.transpose(A, 1, 0) 
                    A = torch.mm(torch.nn.functional.softmax(A, dim=1), h) 
                    emb = model.atten_model.fc_2(A) 
                    
                    t_val = float(r['survival_months'])
                    c_val = int(r['censor']) 
                    e_val = 1.0 if c_val == 0 else 0.0
                        
                    X.append(emb)
                    T.append(t_val)
                    E.append(e_val)
                except: pass
        
        X_t = torch.cat(X)
        T_t = torch.tensor(T, dtype=torch.float32).cuda().view(-1)
        
        E_t = torch.tensor(E, dtype=torch.float32).cuda().view(-1)
        
        print(f"[{split_name}] Shapes -> X: {X_t.shape}, T: {T_t.shape}, E: {E_t.shape}")
        return X_t, T_t, E_t

    print("Extracting features for full-batch Cox...")
    X_tr, T_tr, E_tr = get_reps(df_tr, "Train")
    X_val, T_val, E_val = get_reps(df_val, "Val")
    
    optimizer = optim.AdamW(list(model.fc_3.parameters())+list(model.fc_4.parameters()), lr=1e-4, weight_decay=1e-3)
    stop = EarlyStopping(patience=50, path=os.path.join(args.save_dir, 'phase7_cox.pth'))
    
    print("Starting Cox Fine-tuning...")
    for epoch in range(1000):
        model.fc_3.train(); model.fc_4.train()
        optimizer.zero_grad()
        
        risk = model.fc_4(model.fc_3(X_tr)).squeeze()
        
        if torch.isnan(risk).any():
            print("NaN in risk scores. Stopping.")
            break

        loss = neg_partial_log_likelihood(risk, E_tr.bool(), T_tr)
        
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            vrisk = model.fc_4(model.fc_3(X_val)).squeeze()
            # Validation Step
            vl = neg_partial_log_likelihood(vrisk, E_val.bool(), T_val).item()
            
        if epoch % 100 == 0: 
            print(f"Ep {epoch} Val Loss: {vl:.4f}")
        stop(vl, model)
        if stop.early_stop: break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--feature_dir', required=True)
    parser.add_argument('--phase6_model', required=True)
    parser.add_argument('--save_dir', required=True)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)