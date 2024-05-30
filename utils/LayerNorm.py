from torch import Tensor
from torch.nn import Parameter
from torch_geometric.typing import OptTensor
from torch_scatter import scatter
# from torch_geometric.utils import scatter

from utils.funcs import degree
import torch


class LayerNorm(torch.nn.Module):
    def __init__(self, in_channels, eps=1e-5, affine=True):
        super().__init__()

        self.in_channels = in_channels
        self.eps = eps

        if affine:
            self.weight = Parameter(torch.empty((in_channels,)))
            self.bias = None
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()
        
    def item(exp):
        exp=exp.item()
        return 1/exp
        

    def reset_parameters(self):
        with torch.no_grad():
            if self.weight.size(0) >= 256:
                self.weight.data.fill_(0.5)
            else:
                self.weight.data.fill_(1.)

    def forward(self, x: Tensor, batch: OptTensor = None) -> Tensor:
        """"""
        if batch is None:
            out = x / (x.std(unbiased=False) + self.eps)

        else:
            batch_size = int(batch.max()) + 1

            norm = degree(batch, batch_size, dtype=x.dtype).clamp_(min=1)
            norm = norm.mul_(x.size(-1)).view(-1, 1)

            var = scatter(x * x, batch, dim=0, dim_size=batch_size,
                          reduce='add').sum(dim=-1, keepdim=True)
            var = var / norm

            out = x / (var + self.eps).sqrt().index_select(0, batch)

        if self.weight is not None:
            out = out * self.weight

        return out