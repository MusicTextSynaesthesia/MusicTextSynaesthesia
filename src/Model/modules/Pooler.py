import torch
from torch import nn


class LSTMPooler(nn.Module):
    def __init__(self, dim, pool_size):
        super(LSTMPooler, self).__init__()
        self.dim = dim
        self.pooler = nn.ModuleList([nn.LSTM(dim, dim // 2, batch_first=True, bidirectional=True) for _ in range(pool_size)])

    def forward(self, x):
        x = [m(x)[1][0].transpose(0, 1).view(-1, 1, self.dim) for m in self.pooler]
        x = torch.cat(x, dim=1)
        return x


class AttentionPooler(nn.Module):
    def __init__(self, dim, pool_size):
        super(AttentionPooler, self).__init__()
        self.pool_size = pool_size
        self.embed = nn.Embedding(self.pool_size, dim)
        self.pooler = nn.MultiheadAttention(dim, 8)

    def forward(self, x):
        n = x.shape[0]
        q = self.embed(torch.arange(self.pool_size).repeat(n, 1).cuda())
        q, x = q.transpose(0, 1), x.transpose(0, 1)
        x = self.pooler(q, x, x)[0].transpose(0, 1)
        return x


class LinearPooler(nn.Module):
    def __init__(self, dim, pool_size):
        super(LinearPooler, self).__init__()
        self.pool_size = pool_size
        self.pooler = nn.Linear(dim, pool_size*dim)

    def forward(self, x):
        n, _, dim = x.shape
        x = self.pooler(x.mean(dim=1))
        return x.view(n, self.pool_size, dim)


class MeanPooler(nn.Module):
    def __init__(self, dim, pool_size):
        super(MeanPooler, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        pool = self.pool_size
        split = [x.shape[1] // pool + (1 if i < x.shape[1] % pool else 0) for i in range(pool)]
        x = [y.mean(dim=1, keepdim=True) for y in x.split(split, dim=1)]
        return torch.cat(x, dim=1)
