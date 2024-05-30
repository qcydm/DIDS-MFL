import torch
from utils.Implicit_Func import Implicit_Func

class Implicit_Module(torch.nn.Module):
    def __init__(self, hidden_channel, middle_channels, alpha, norm, dropout, act, double_linear, rescale):
        super().__init__()
        Fs = [Implicit_Func(hidden_channel, middle_channel, alpha, norm, dropout, act, double_linear, rescale) for
              middle_channel in middle_channels]
        self.Fs = torch.nn.ModuleList(Fs)

    def _reset(self, z):
        for func in self.Fs:
            func._reset(z)

    def forward(self, z, x, edge_index, norm_factor, batch):
        for func in self.Fs:
            z = func(z, x, edge_index, norm_factor, batch)
        return z
