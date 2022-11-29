import torch


def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def koza_leon(x, k=1):
    alignment = x @ x.T - torch.eye(x.shape[0]) # eliminate diag elements
    max_alignment = torch.max(distances, dim=0)
    min_distances = 2*(1 - max_alignment) 
    return - torch.sum(torch.log(min_distances)) # minus sign to maximize

def inv_loss(x, y, alpha=2):
    return (x @ y.T).mean()

__all__ = ['align_loss', 'uniform_loss']
