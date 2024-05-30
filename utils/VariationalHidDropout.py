import torch


class VariationalHidDropout(torch.nn.Module):
    def __init__(self, dropout=0.0):
        super(VariationalHidDropout, self).__init__()
        self.dropout = dropout
        self.mask = None

    def reset_mask(self, z):
        m = torch.zeros_like(z).bernoulli_(1 - self.dropout)

        mask = m.requires_grad_(False) / (1 - self.dropout)
        self.mask = mask
        return mask

    def forward(self, x):
        if not self.training or self.dropout == 0:
            return x
        assert self.mask is not None, f"You need to reset mask before using {self.__class__.__name__}"
        assert self.mask.size() == x.size()  # Make sure the dimension matches
        return self.mask * x
