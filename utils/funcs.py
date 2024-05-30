from collections import Counter
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn.functional as F
from scipy.sparse.linalg import svds
from torch_geometric.data import TemporalData
from torch_geometric.utils import to_undirected, degree
from torch_scatter import scatter_add
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from utils.LayerNorm import LayerNorm


def nodeMap(edge_index, mode='encode', decode_dict=None):
    if mode == 'encode':
        src, dst = edge_index.tolist()
        nodeSet = sorted(list(set(src + dst)))
        assoc = list(range(0, len(nodeSet)))
        m = [dict(zip(nodeSet, assoc)), dict(zip(assoc, nodeSet))]
        src = [m[0][i] for i in src]
        dst = [m[0][i] for i in dst]
        edge_index = torch.stack((torch.tensor(src), torch.tensor(dst)), dim=0)
        return edge_index, m
    elif mode == 'decode':
        src, dst = edge_index.tolist()
        src = [decode_dict[i] for i in src]
        dst = [decode_dict[i] for i in dst]
        edge_index = torch.stack((torch.tensor(src), torch.tensor(dst)), dim=0)
        return edge_index
    else:
        print('Error mode.')


def get_act(act_type):
    act_type = act_type.lower()
    if act_type == 'identity':
        return torch.nn.Identity()
    if act_type == 'relu':
        return torch.nn.ReLU(inplace=True)
    elif act_type == 'elu':
        return torch.nn.ELU(inplace=True)
    elif act_type == 'tanh':
        return torch.nn.Tanh()
    elif act_type == 'sigmoid':
        return torch.nn.LogSigmoid()
    else:
        raise NotImplementedError


def cal_norm(edge_index, num_nodes=None, self_loop=False, cut=False):
    # calculate normalization factors: (2*D)^{-1/2}
    if num_nodes is None:
        num_nodes = edge_index.max() + 1
    D = degree(edge_index[0], num_nodes)
    if self_loop:
        D = D + 1

    if cut:  # for symmetric adj
        D = torch.sqrt(1 / D)
        D[D == float("inf")] = 0.
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        row, col = edge_index
        mask = row < col
        edge_index = edge_index[:, mask]
    else:
        D = torch.sqrt(1 / 2 / D)
        D[D == float("inf")] = 0.
    # D = Tensor([0.5] * num_nodes).to(device)
    if D.dim() == 1:
        D = D.unsqueeze(-1)
    return D, edge_index


@torch.enable_grad()
def regularize(z, x, reg_type, edge_index=None, norm_factor=None):
    z_reg = norm_factor * z

    if reg_type == 'Lap':  # Laplacian Regularization
        row, col = edge_index
        loss = scatter_add(((z_reg.index_select(0, row) - z_reg.index_select(0, col)) ** 2).sum(-1), col, dim=0,
                           dim_size=z.size(0))
        return loss.mean()

    elif reg_type == 'Dec':  # Feature Decorrelation
        zzt = torch.mm(z_reg.t(), z_reg)
        Dig = 1. / torch.sqrt(1e-8 + torch.diag(zzt, 0))
        z_new = torch.mm(z_reg, torch.diag(Dig))
        zzt = torch.mm(z_new.t(), z_new)
        zzt = zzt - torch.diag(torch.diag(zzt, 0))
        zzt = F.hardshrink(zzt, lambd=0.5)
        square_loss = F.mse_loss(zzt, torch.zeros_like(zzt))
        return square_loss

    else:
        raise NotImplementedError

def filtered_data(data,num):
    # data = torch.load("./data/CIC-ToN-IoT.pt")
    # print(data)
    unique_attacks = data['attack'].unique()
    selected_attacks = unique_attacks[torch.randperm(len(unique_attacks))[:num]]
    other_attacks_mask = ~torch.isin(unique_attacks, selected_attacks)
    other_attacks = unique_attacks[other_attacks_mask]

    counts = Counter(data['attack'][torch.isin(data['attack'], selected_attacks)].tolist())
    counts_other = Counter(data['attack'][torch.isin(data['attack'], other_attacks)].tolist())
    sorted_attacks = sorted(counts, key=counts.get, reverse=True)
    sorted_other = sorted(counts_other, key=counts_other.get, reverse=True)

    attack_to_new_index = {attack: idx for idx, attack in enumerate(sorted_attacks)}
    attack_to_other_index = {attack: idx for idx, attack in enumerate(sorted_other)}
    selected_indices = torch.where(torch.isin(data['attack'], selected_attacks))[0]
    other_indices = torch.where(~torch.isin(data['attack'], selected_attacks))[0]

    def create_temporal_data(data, indices, attack_to_index):
        new_attack_indices = torch.tensor([attack_to_index[a.item()] for a in data['attack'][indices]], dtype=torch.long)
        return TemporalData(
            src=data['src'][indices],
            dst=data['dst'][indices],
            t=data['t'][indices],
            msg=data['msg'][indices],
            src_layer=data['src_layer'][indices],
            dst_layer=data['dst_layer'][indices],
            dt=data['dt'][indices],
            label=data['label'][indices],
            attack=new_attack_indices
        )

    filtered_data_selected = create_temporal_data(data, selected_indices, attack_to_new_index)
    other_data_selected = create_temporal_data(data, other_indices, attack_to_other_index)

    # torch.save(filtered_data_selected, f"{save_path}_selected.pt")
    # torch.save(other_data_selected, f"{save_path}_other.pt")
    # print(filtered_data_selected)
    return filtered_data_selected, other_data_selected

def enhance_sim_matrix(C, K, d, alpha):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5 * (C + C.T)
    r = min(d * K + 1, C.shape[0] - 1)
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = 0.5 * (L + L.T)
    L = L / L.max()
    return L


def repeat_column(matrix, j):
    num_columns = matrix.shape[1]
    if j < 0 or j >= num_columns:
        raise ValueError("Invalid column index")
    column = matrix[:, j]
    repeated_column = np.tile(column, (1, num_columns - 1))
    return repeated_column


def delta2(x):
    Z = torch.zeros(x.shape[0], x.shape[0])
    for j in range(x.shape[1]):
        delta = repeat_column(x, j) - torch.cat([x[:, :j], x[:, j + 1:]], dim=1)
        Z += torch.mm(delta, delta.T)
    return Z


# Repeat a specific column of a tensor
def repeat_column(tensor, col_index):
    return tensor[:, col_index].unsqueeze(1).repeat(1, tensor.shape[1]-1)



def compute_Q_star(Z, P, gamma, Phi1, Phi2, mu, Q_tilde):
    # Calculate Delta3
    Delta3 = P.t() @ Z + Phi1 / mu

    # Calculate the numerator part
    numerator = mu * gamma *Delta3  @ Z.t() @ P + mu * Q_tilde - Phi2

    # Calculate the denominator part
    denominator = mu * gamma ** 2 *  P.t() @ Z @Z.t() @ P + mu * torch.eye(Z.shape[0])
        
    # Invert the denominator and calculate Q_star
    Q_star = (torch.inverse(denominator) @ numerator)*0.0001

    return Q_star
  


def compute_tilde_Q_star(Q, Phi2, mu):
    # Calculate Q + Phi2 / mu
    Q_plus_Phi2_over_mu = Q + Phi2 / mu

    # Perform singular value decomposition
    U, Sigma, Vt = torch.svd(Q_plus_Phi2_over_mu)

    # Calculate the S function
    S = torch.matmul(U, torch.matmul(
        torch.diag(torch.sign(Sigma) * torch.maximum(torch.abs(Sigma - 1 / mu), torch.tensor(0.0))), Vt))

    return S


def MFL(x, C, gamma, Eta, alpha):
    phi1 = torch.randn((100,128))
    phi2 = torch.zeros((100,100))
    miu = 0.1
    Z = x.cpu()
    miu_max = 1e7
    Z_t = x * gamma
    Q = torch.zeros_like(torch.tensor(C))
    Q_tilde = Q
    i = 0
    result=None
    mu_decay = 0.9
    layer=LayerNorm
    while True:
        delta1 = Z * (1 - gamma)
        delta3 = Z - gamma * Q @ Z
        i+=1
        x1 = alpha * delta1 @ delta1.t()
        x2 = Eta * delta2(Z)
        x3 = miu * delta3 @ delta3.t()
        x4 = torch.inverse(x1 - x2 + x3 + torch.eye(Z.shape[0]))
        P = - phi1 @ (delta3.T @ x4)
        Q = compute_Q_star(Z, P, gamma, phi1, phi2, miu, Q_tilde=Q_tilde)
        # Q_tilde needs to be updated
        Q_tilde = compute_tilde_Q_star(Q, phi2, miu)
        if result is None:
            result = 0.5 * (Q + Q.t())
        phi1 = phi1 + miu * (P.t() @ Z - gamma * P.t() @ Q @ Z)
        phi2 = phi2 + miu * (Q - Q_tilde)
        miu = min(1.01 * miu * mu_decay, miu_max)
        # x = Z @ P
        # y = Z @ P - gamma * Q @ Z @ P
        expression_value = torch.norm((P @ Z - gamma * P @ Q  @ Z) , float('inf'))
        expression_final=layer.item(expression_value)

        if expression_final < 1e-6:
            break
    return result