import torch
from utils.funcs import get_act
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, c_in, c_out, middle_channels, hidden_act='sigmoid', out_act='identity', dropout=0.):
        super(MLP, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.middle_channels = middle_channels

        self.hidden_act = get_act(hidden_act)
        self.out_act = get_act(out_act)

        c_ins = [c_in] + middle_channels
        c_outs = middle_channels + [c_out]

        self.lins = torch.nn.ModuleList()
        for _, (in_dim, out_dim) in enumerate(zip(c_ins, c_outs)):
            self.lins.append(torch.nn.Linear(int(in_dim), int(out_dim)))

        self.drop = torch.nn.Dropout(dropout) if dropout > 0. else torch.nn.Identity()

    def forward(self, xs):
        if len(self.lins) > 1:
            for _, lin in enumerate(self.lins[:-1]):
                xs = lin(xs)
                xs = self.hidden_act(xs)
                xs = self.drop(xs)
            xs = self.lins[-1](xs)
            xs = self.out_act(xs)

        else:
            xs = self.drop(xs)
            xs = self.lins[-1](xs)

        return xs

class MLPPredictor(torch.nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = torch.nn.Linear(in_features * 2, out_classes)

    def forward(self, z_src, z_dst):
        score = self.W(torch.cat([z_src, z_dst], 1))
        return score
    
    
    
class DRMLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DRMLP, self).__init__()
        self.relu=torch.nn.ReLU(inplace=False)
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class FeatureTransformer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureTransformer, self).__init__()
        self.transformation = torch.nn.Linear(input_dim, output_dim)
        self.tanh=torch.tanh
    
    def forward(self, x):
        return torch.tanh(self.transformation(x))