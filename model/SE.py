import torch
from torch.nn.parameter import Parameter
from torch.nn import Module

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SelfExpr(Module):
    def __init__(self, n):
        self.n = n
        super(SelfExpr, self).__init__()
        # self.weight = Parameter(1*torch.FloatTensor(n, n))
        self.weight = Parameter(torch.FloatTensor(n, n).uniform_(0, 0.01).to(device))

    def forward(self, input):
        # self.weight.data = F.relu(self.weight)
        output = torch.mm(self.weight - torch.diag(torch.diagonal(self.weight)), input)
        return self.weight.to(device), output.to(device)

    def reset(self):
        self.weight.data = torch.FloatTensor(self.n, self.n).uniform_(0, 0.01).to(device)
