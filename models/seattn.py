import torch
import torch.nn as nn
import torch.nn.functional as F

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=512, D=256, dropout=0.2, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]

        if dropout:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))
        
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)
    
    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x

class MIL_attn(nn.Module):
    """
    Handles Batch Size = 1. Input: [patch_size, input_dim]
    """
    def __init__(self, L=512, D=256, dropout=0.2, n_classes=1, input_dim=2048, attention_only=False):
        super(MIL_attn, self).__init__()
        self.attention_only = attention_only
        self.n_classes = n_classes
        self.fc_1 = nn.Sequential(*[nn.Linear(input_dim, L), nn.ReLU(), nn.Dropout(dropout)])
        self.attn_net = Attn_Net_Gated(L=L, D=D, dropout=dropout, n_classes=n_classes)
        self.fc_2 = nn.Sequential(*[nn.Linear(L, D), nn.ReLU(), nn.Dropout(dropout)])
        self.classifier = nn.Linear(D, n_classes)
        
    def forward(self, x):
        x = self.fc_1(x)
        A, h = self.attn_net(x)
        A = torch.transpose(A, 1, 0)
        if self.attention_only: return A
        A = torch.mm(F.softmax(A, dim=1), h)
        A = self.fc_2(A)
        logits = self.classifier(A)
        if logits.dim() == 1: logits = logits.view(1, -1)
        return logits

class MIL_attn_2(nn.Module):
    """
    Handles Batch Size > 1. Input: [batch_size, patch_size, input_dim]
    """
    def __init__(self, L=512, D=256, dropout=0.2, n_classes=1, input_dim=2048, attention_only=False):
        super(MIL_attn_2, self).__init__()
        self.attention_only = attention_only
        self.n_classes = n_classes
        self.fc_1 = nn.Sequential(*[nn.Linear(input_dim, L), nn.ReLU(), nn.Dropout(dropout)])
        self.attn_net = Attn_Net_Gated(L=L, D=D, dropout=dropout, n_classes=n_classes)
        self.fc_2 = nn.Sequential(*[nn.Linear(L, D), nn.ReLU(), nn.Dropout(dropout)])
        self.classifier = nn.Linear(D, n_classes)
        
    def forward(self, x):
        x = self.fc_1(x)
        A, h = self.attn_net(x)
        A = torch.transpose(A, 2, 1)
        if self.attention_only: return A
        A = torch.matmul(F.softmax(A, dim=2), h)
        A = self.fc_2(A)
        logits = self.classifier(A).squeeze(1)
        return logits

class ResNet_MIL(nn.Module):
    def __init__(self, n_classes=1, dropout=0.25, L=512, D=256, freeze_blocks_1_3=False):
        super(ResNet_MIL, self).__init__()
        import torchvision.models as models
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        if freeze_blocks_1_3:
            for param in resnet.parameters(): param.requires_grad = False
            for param in resnet.layer4.parameters(): param.requires_grad = True
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1]) 
        self.attn_model = MIL_attn(L=L, D=D, n_classes=n_classes, dropout=dropout)

    def forward(self, x):
        x = x.squeeze(0) 
        feats = self.feature_extractor(x).flatten(1)
        logits = self.attn_model(feats)
        return logits
