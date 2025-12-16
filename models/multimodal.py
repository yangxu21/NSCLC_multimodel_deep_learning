import torch
import torch.nn as nn


# Building Blocks
def MLP_block(dim1, dim2, dropout=0.25):
    """Standard MLP block: Linear -> ReLU -> Dropout."""
    return nn.Sequential(nn.Linear(dim1, dim2), nn.ReLU(), nn.Dropout(dropout))

def SNN_block(dim1, dim2, dropout=0.25):
    """Self-Normalizing Neural Network block: Linear -> SELU -> AlphaDropout."""
    return nn.Sequential(nn.Linear(dim1, dim2), nn.SELU(), nn.AlphaDropout(dropout))

class FeatureEncoder(nn.Module):
    """
    Encodes a specific modality (e.g., NGS, Clinical, or WSI vectors) 
    into a latent representation.
    """
    def __init__(self, input_dim, sizes=[256, 128], block_type='MLP', dropout=0.25):
        super(FeatureEncoder, self).__init__()
        
        # Select architecture based on feature density/type
        if block_type == 'MLP':
            block = MLP_block
        else:
            block = SNN_block
            
        layers = [block(input_dim, sizes[0], dropout)]
        for i in range(len(sizes)-1):
            layers.append(block(sizes[i], sizes[i+1], dropout))
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = sizes[-1]

    def forward(self, x):
        return self.encoder(x)


# Individual Model (For Step 1)
class IndividualCoxModel(nn.Module):
    """
    Wraps an encoder with a final linear layer for Cox regression.
    Used for Step 1: Training individual modalities (e.g. Mutation only).
    """
    def __init__(self, input_dim, sizes=[256, 128], block_type='MLP', dropout=0.25):
        super(IndividualCoxModel, self).__init__()
        self.encoder = FeatureEncoder(input_dim, sizes, block_type, dropout)
        self.fc_out = nn.Linear(self.encoder.output_dim, 1) # Output 1 log_hazard
        
    def forward(self, x):
        x = self.encoder(x)
        return self.fc_out(x)


# Fusion Model (For Step 2)
class MMDL_Fusion(nn.Module):
    """
    Multi-Modal Deep Learning Fusion Model.
    Architecture:
        1. Individual Encoders for each modality.
        2. Concatenation of latent embeddings.
        3. Fusion MLP to predict Cox Hazard Ratio.
    """
    def __init__(self, block_type='MLP'):
        super(MMDL_Fusion, self).__init__()
        
        # 1. WSI Branch
        self.wsi_net = FeatureEncoder(input_dim=512, sizes=[256, 128], block_type=block_type)
        # 2. NGS & Clinical Branches (Input dims based on dataset)
        self.cna_amp  = FeatureEncoder(100, [256, 128], block_type='SNN')
        self.cna_del  = FeatureEncoder(100, [256, 128], block_type='SNN')
        self.mut_data = FeatureEncoder(100, [256, 128], block_type='MLP')
        self.amp_set  = FeatureEncoder(100, [256, 128], block_type='MLP')
        self.del_set  = FeatureEncoder(100, [256, 128], block_type='MLP')
        self.mut_set  = FeatureEncoder(100, [256, 128], block_type='MLP')
        self.clinical = FeatureEncoder(5, [64, 32], block_type='MLP')

        # 3. Fusion Layer
        fusion_in = (128 * 7) + 32
        
        self.fusion = nn.Sequential(
            MLP_block(fusion_in, 256),
            MLP_block(256, 256),
            nn.Linear(256, 1) # Output: Log Hazard Ratio
        )

    def forward(self, wsi, cna_amp, cna_del, mut_data, amp_set, del_set, mut_set, clinical):
        """
        Forward pass expecting batch tensors.
        wsi shape: (Batch, 512)
        """
        # Encode individual modalities
        w_emb = self.wsi_net(wsi)      
        c_amp = self.cna_amp(cna_amp)   
        c_del = self.cna_del(cna_del)   
        m_dat = self.mut_data(mut_data) 
        a_set = self.amp_set(amp_set)   
        d_set = self.del_set(del_set)   
        m_set = self.mut_set(mut_set)   
        clin  = self.clinical(clinical) 
        
        # Concatenate embeddings
        combined = torch.cat([w_emb, c_amp, c_del, m_dat, a_set, d_set, m_set, clin], dim=1)
        
        # Final prediction
        return self.fusion(combined)