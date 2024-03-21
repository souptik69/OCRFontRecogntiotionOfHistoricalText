import torch
from torch.nn import functional as F


class SmoothCE(torch.nn.Module):

    def __init__(self, eps=0.4, trg_pad_idx=0, reduction='sum'):
        super(SmoothCE, self).__init__()
        self.eps = eps
        self.trg_pad_idx = trg_pad_idx
        self.reduction = reduction

    def forward(self, pred, gold):
        pred = pred.reshape(-1,pred.shape[-1])
        gold = gold.contiguous().view(-1)

        if self.eps > 0.:
            n_class = pred.size(1)
            v = gold.view(-1,1)
            one_hot = torch.zeros_like(pred)
            one_hot = one_hot.scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            non_pad_mask = gold.ne(self.trg_pad_idx)
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask)       # average later

        else:
            loss = F.cross_entropy(pred, gold, ignore_index=self.trg_pad_idx, reduction='none')
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction=='sum':
            return loss.sum()
        elif self.reduction == None:
            return loss