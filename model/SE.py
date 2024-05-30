import torch
from torch.nn.parameter import Parameter
from torch.nn import Module


class SelfExpr(Module):
    def __init__(self, n):
        self.n = n
        super(SelfExpr, self).__init__()
        # self.weight = Parameter(1*torch.FloatTensor(n, n))
        self.weight = Parameter(torch.FloatTensor(n, n).uniform_(0, 0.01))

    def forward(self, input):
        # self.weight.data = F.relu(self.weight)
        output = torch.mm(self.weight - torch.diag(torch.diagonal(self.weight)), input)
        return self.weight, output

    def reset(self):
        self.weight.data = torch.FloatTensor(self.n, self.n).uniform_(0, 0.01)