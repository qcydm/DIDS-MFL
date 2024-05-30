from utils.VariationalHidDropout import VariationalHidDropout
from utils.funcs import *
from utils.LayerNorm import LayerNorm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Implicit_Func(torch.nn.Module):
    def __init__(self, hidden_channel, middle_channel, alpha, norm, dropout, act, double_linear, rescale):
        super().__init__()
        self.alpha = alpha
        self.W = torch.nn.Linear(hidden_channel, hidden_channel, bias=False)

        self.double_linear = double_linear
        if self.double_linear:
            self.U = torch.nn.Linear(hidden_channel, middle_channel)

        self.norm = eval(norm)(middle_channel)

        self.rescale = rescale

        self.act = get_act(act)

        self.drop = VariationalHidDropout(dropout)

    def _reset(self, z):
        self.drop.reset_mask(z)

    def forward(self, z, x, edge_index, norm_factor, batch):
        num_nodes = x.size(0)
        row, col = edge_index

        if self.rescale:
            degree = 1. / norm_factor
            degree[degree == float("inf")] = 0.
        else:
            degree = 1.
        degree = degree.to(device)
        # z.to(device)
        # x.to(device)
        if self.double_linear:
            # print(z)
            # print(x)
            # print(self.W(z))
            # print(self.U(x))
            WzUx = self.W(z) + degree * self.U(x)
        else:
            WzUx = self.W(z + degree * x)

        WzUx = norm_factor * WzUx
        WzUx = WzUx.index_select(0, row) - WzUx.index_select(0, col)

        if batch is not None:
            WzUx = self.norm(self.act(WzUx), batch.index_select(0, row))
        else:
            WzUx = self.norm(self.act(WzUx))

        new_z = scatter_add(WzUx * norm_factor[row], row, dim=0, dim_size=num_nodes)
        new_z -= scatter_add(WzUx * norm_factor[col], col, dim=0, dim_size=num_nodes)

        new_z = -F.linear(new_z, self.W.weight.t())

        z = self.alpha * self.drop(new_z) + (1 - self.alpha) * z

        return z