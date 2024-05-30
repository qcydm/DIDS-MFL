from torch import autograd

from utils.funcs import *


class Append_func(torch.nn.Module):
    def __init__(self, coeff, reg_type):
        super().__init__()
        self.coeff = coeff
        self.reg_type = reg_type

    def forward(self, z, x, edge_index, norm_factor):
        if self.reg_type == '' or self.coeff == 0.:
            return z
        else:
            z = z if z.requires_grad else z.clone().detach().requires_grad_()
            reg_loss = regularize(z, x, self.reg_type, edge_index, norm_factor)
            grad = autograd.grad(reg_loss, z, create_graph=True)[0]
            z = z - self.coeff * grad
            return z