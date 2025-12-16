import torch
import torch.nn as nn

class NLLSurvLoss(nn.Module):
    '''
    The negative log-likelihood loss function (Zadeh and Schmid, 2021).
    '''
    def __init__(self, alpha=0.0, eps=1e-7, reduction='mean'):
        super(NLLSurvLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction
    def forward(self, hazards, y, c):
        return nll_loss(hazards=hazards, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1),
                        alpha=self.alpha, eps=self.eps,
                        reduction=self.reduction)

def nll_loss(hazards, y, c, alpha=0.0, eps=1e-7, reduction='mean'):
    """
    y: (n_batches, 1); The true time bin index label.
    c: (n_batches, 1); The censoring status, with 1 indicating censoring.
    """
    y = y.type(torch.int64)
    c = c.type(torch.int64)
    S = torch.cumprod(1 - hazards, dim=1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)
    
    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)
    neg_l = censored_loss + uncensored_loss
    
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))
    return loss