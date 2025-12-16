import torch
import torch.nn as nn
from models.seattn import MIL_attn, MIL_attn_2

class discrete_time_to_event_model(nn.Module):
    def __init__(self, n_classes=4, n_features=[256, 32], dropout=0.25):
        super(discrete_time_to_event_model, self).__init__()
        self.fc_3 = nn.Sequential(nn.Linear(n_features[0], n_features[1]), nn.ReLU(), nn.Dropout(dropout))
        self.fc_4 = nn.Sequential(nn.Linear(n_features[1], n_classes))
    def forward(self, x):
        x = self.fc_3(x)
        logits = self.fc_4(x) 
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S, Y_hat

class attention_discrete_time_survival_model(nn.Module):
    def __init__(self, n_classes=4, n_features=[256, 32], dropout=0.25):
        super(attention_discrete_time_survival_model, self).__init__()
        # Use MIL_attn (Batch=1)
        self.atten_model = MIL_attn(attention_only=False, dropout=dropout, n_classes=1)
        self.atten_model.classifier = nn.Identity()
        self.fc_3 = nn.Sequential(nn.Linear(n_features[0], n_features[1]), nn.ReLU(), nn.Dropout(dropout))
        self.fc_4 = nn.Sequential(nn.Linear(n_features[1], n_classes))
        
    def forward(self, x):
        x = self.atten_model(x) # [1, 256]
        x = self.fc_3(x)
        logits = self.fc_4(x)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S, Y_hat

class attention_COX_model(nn.Module):
    def __init__(self, n_classes=1, n_features=[256, 32], dropout=0.25):
        super(attention_COX_model, self).__init__()
        # Use MIL_attn_2 (Batch > 1)
        self.atten_model = MIL_attn_2(attention_only=False, dropout=dropout, n_classes=1)
        self.atten_model.classifier = nn.Identity()
        self.fc_3 = nn.Sequential(nn.Linear(n_features[0], n_features[1]), nn.ReLU(), nn.Dropout(dropout))
        self.fc_4 = nn.Sequential(nn.Linear(n_features[1], n_classes))
        
    def forward(self, x):
        x = self.atten_model(x)
        x = self.fc_3(x)
        log_hz = self.fc_4(x)
        return log_hz