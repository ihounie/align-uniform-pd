import torch


def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2, mode = "all", k = 12):
    if mode=="all":
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
    else:
        alignment = x @ x.T * (torch.eye(x.shape[0], device=x.device)==0) # eliminate diag elements
        max_alignment, _ = torch.sort(alignment, dim=0, descending=True) 
        if mode=="only_k":
            return (2*(1-max_alignment[k,:])).mul(-t).exp().mean().log()
        elif mode=="mean_k":
            return (2*(1-max_alignment[:k,:])).mul(-t).exp().mean().log()
        else:
            raise NotImplementedError

def koza_leon(x, k=10, t=2):
    alignment = x @ x.T * (torch.eye(x.shape[0], device=x.device)==0) # eliminate diag elements
    max_alignment, _ = torch.sort(alignment, dim=0, descending=True)
    return (2*(1-max_alignment[k,:])).mul(-t).exp().mean().log()

def inv_loss(x, y, alpha=2):
    return (x @ y.T).mean()

__all__ = ['align_loss', 'uniform_loss']
