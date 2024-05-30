import time

import numpy as np
import random
import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
from torch_geometric.loader import TemporalDataLoader
from sklearn.preprocessing import normalize
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)

from model.MGD import *
from utils.LOSS import Loss
from utils.MLP import MLPPredictor
from utils.funcs import *
from sklearn.metrics import f1_score
from sklearn.metrics import normalized_mutual_info_score

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_all = torch.load('./data/CIC-BoT-IoT.pt')
# 假设 data 是您的数据集对象
print(data_all)
class_pre = 2
class_meta_train = 3
class_meta_test = 3
data1,data = filtered_data(data_all, class_meta_train)
data,_ = filtered_data(data_all, class_pre)

# data = torch.load("./data/filtered_-306-789_CIC-ToN-IoT.pt")
# data1 = torch.load("./data/filtered_-306_CIC-ToN-IoT.pt")
# print(data)
data1 = data1.to(device)
data = data.to(device)
# min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())
neighbor_loader = LastNeighborLoader(data.num_nodes+data1.num_nodes, size=20, device=device)
# neighbor_loader1 = LastNeighborLoader(data1.num_nodes, size=20, device=device)
memory_dim = time_dim = embedding_dim = 64

train_loader = TemporalDataLoader(data, batch_size=50)
layer = 1
gind_params = {'num_layers': 1, 'alpha': 0.02, 'hidden_channels': 64, 'drop_input': True, 'dropout_imp': 0.5,
               'dropout_exp': 0.0, 'iter_nums': [36, 4], 'linear': True, 'double_linear': True, 'act_imp': 'tanh',
               'act_exp': 'elu', 'rescale': True, 'residual': True, 'norm': 'LayerNorm', 'final_reduce': None}

memory = TGNMemory(
    data.num_nodes+data1.num_nodes,
    data.msg.size(-1),
    memory_dim,
    time_dim,
    message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
    aggregator_module=LastAggregator(),
).to(device)

# memory1 = TGNMemory(
#     data1.num_nodes,
#     data1.msg.size(-1),
#     memory_dim,
#     time_dim,
#     message_module=IdentityMessage(data1.msg.size(-1), memory_dim, time_dim),
#     aggregator_module=LastAggregator(),
# ).to(device)


mgd = MGD(in_channels=embedding_dim, out_channels=embedding_dim, **gind_params).to(device)

bin_predictor = MLPPredictor(in_features=embedding_dim, out_classes=2).to(device)
mul_predictor = MLPPredictor(in_features=embedding_dim, out_classes=5).to(device)
# model1 = myMLP(in_channels=256,hidden_channels=128,out_channels=256).to(device)
# optimizer = torch.optim.Adam(set(memory1.parameters()) | set(mgd.parameters()) | set(memory1.parameters()), lr=0.0001)
optimizer = torch.optim.Adam(set(memory.parameters()) | set(mgd.parameters()), lr=0.0001)
# optimizer1 = torch.optim.Adam(set(model1.parameters()), lr=0.0001)
criterion = Loss(2, 5)
assoc = torch.empty(data1.num_nodes+data.num_nodes, dtype=torch.long, device=device)

def sample_train(datax, k, num_query, split_ratio):
    classes = [0, 1, 2]
    indices_support = []
    indices_query = []

    for i in classes:
        # 获取当前类别的所有样本索引
        class_indices = np.where(datax.attack == i)[0]

        # 计算当前类别样本的总数
        num_class_samples = len(class_indices)

        # 根据比例计算当前类别的划分点
        class_split_index = int(num_class_samples * split_ratio)

        # 只考虑当前类别中按比例划分的样本
        class_indices = class_indices[:class_split_index]

        # 从当前类别的样本中随机选择支持集和查询集
        if class_split_index > 0:  # 确保有足够的样本进行划分
            random_selection = np.random.choice(class_indices, size=k + num_query, replace=False)
            selected_list = random_selection.tolist()
            indices_support.extend(selected_list[:k])
            indices_query.extend(selected_list[k:])

    # 使用支持集和查询集的索引来抽取数据
    data_sup = datax[indices_support]
    data_q = datax[indices_query]

    return data_sup, data_q

def sample_test(datax, k, num_query, split_ratio):
    classes = [0, 1, 2]
    indices_support = []
    indices_query = []

    for i in classes:
        # 获取当前类别的所有样本索引
        class_indices = np.where(datax.attack == i)[0]

        # 计算当前类别样本的总数
        num_class_samples = len(class_indices)

        # 根据比例计算当前类别的划分点
        class_split_index = int(num_class_samples * split_ratio)

        # 只考虑当前类别中按比例划分的样本
        class_indices = class_indices[class_split_index:]

        # 从当前类别的样本中随机选择支持集和查询集
        if class_split_index > 0:  # 确保有足够的样本进行划分
            random_selection = np.random.choice(class_indices, size=k + num_query, replace=False)
            selected_list = random_selection.tolist()
            indices_support.extend(selected_list[:k])
            indices_query.extend(selected_list[k:])

    # 使用支持集和查询集的索引来抽取数据
    data_sup = datax[indices_support]
    data_q = datax[indices_query]

    return data_sup, data_q

def calculate_subspaces(support_embeddings, num_augmentations, num_classes):
    # Initialize subspaces as a dictionary to store basis for each class
    subspaces = {i: None for i in range(num_classes)}
    
    # 对每个类别进行特征值分解
    for class_idx in range(num_classes):
        # 提取属于当前类别的所有嵌入
        class_embeddings = support_embeddings[class_idx*num_augmentations:(class_idx+1)*num_augmentations, :]
        # 计算协方差矩阵
        covariance_matrix = torch.matmul(class_embeddings.T, class_embeddings) / num_augmentations
        # 执行特征值分解
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
        # 选取最大的特征值对应的特征向量作为子空间的基
        subspace_basis = eigenvectors[:, eigenvalues.argsort(descending=True)]
        subspaces[class_idx] = subspace_basis
        
    return subspaces

def calculate_projection_lengths(query_embeddings, subspaces):
    # 计算投影长度
    # 初始化一个列表，用于保存每个查询样本对于所有类别的投影长度
    projection_lengths = torch.zeros(query_embeddings.shape[0], len(subspaces))

    # 遍历所有类别，计算每个类别的子空间投影长度
    for class_idx, basis in subspaces.items():
        # 投影到当前类别的子空间
        projection = torch.matmul(query_embeddings, basis)
        # 计算到子空间的距离（注意：在这里，我们计算的是距离，而不是长度）
        distances = torch.norm(query_embeddings - projection, dim=1)
        # 保存结果
        projection_lengths[:, class_idx] = distances

    return projection_lengths


def calculate_probabilities(projection_lengths, t=0.1):
    # 从每行中减去最大值来增加数值稳定性
    projection_lengths = projection_lengths - torch.max(projection_lengths, dim=1, keepdim=True)[0]
    exps = torch.exp(projection_lengths / t)
    sum_exps = torch.sum(exps, dim=1, keepdim=True)
    probabilities = exps / sum_exps
    return probabilities


def meta_train():
    memory.train()
    mgd.train()
    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()
    losses = 0
    f1 = 0
    datax, _ = filtered_data(data1, 3)
    for i in range(100):
        support, query = sample_train(datax, 5, 20, split_ratio=0.8)
        optimizer.zero_grad()

        # Prepare support set
        batch = support.to(device)
        src, dst, t, msg, label, attack = batch.src, batch.dst, batch.t, batch.msg, batch.label, batch.attack
        num_classes = torch.max(attack) + 1
        n_id = torch.cat([src, dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        z, last_update = memory(n_id)
        ed, m = nodeMap(torch.stack((src, dst), dim=0))
        ed = ed.to(device)
        norm_factor, ed = cal_norm(ed, num_nodes=len(z), self_loop=False)
        z = mgd(z, ed, norm_factor).to(device)
        support_embeddings = torch.cat([z[assoc[src]], z[assoc[dst]]], 1)
        num_classes = torch.max(attack) + 1
        subspaces = calculate_subspaces(support_embeddings, 3, num_classes)
        # print(support_embeddings.shape)
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)

        # Prepare query set
        batch = query.to(device)
        src, dst, t, msg, label, attack = batch.src, batch.dst, batch.t, batch.msg, batch.label, batch.attack
        n_id = torch.cat([src, dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        z, last_update = memory(n_id)
        ed, m = nodeMap(torch.stack((src, dst), dim=0))
        ed = ed.to(device)
        norm_factor, ed = cal_norm(ed, num_nodes=len(z), self_loop=False)
        z = mgd(z, ed, norm_factor).to(device)
        query_embeddings = torch.cat([z[assoc[src]], z[assoc[dst]]], 1)
        projection_lengths = calculate_projection_lengths(query_embeddings, subspaces)
        # print("Projection lengths shape:", projection_lengths.shape)
        probabilities = calculate_probabilities(projection_lengths)
        true = attack.cpu().numpy()
        # print("Probabilities shape:", probabilities.shape)
        # print(probabilities)
        loss = F.cross_entropy(probabilities, torch.tensor(true))
        loss.backward()
        optimizer.step()
        memory.detach()

        # 将查询样本分类到最近的子空间
        _, preds = torch.max(probabilities, dim=1)
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)
        # print("Predictions shape:", preds.shape)
        # print("True labels shape:", true.shape)
        # Compute accuracy or F1 score
        test_f1 = f1_score(true, preds.cpu().numpy(), average='weighted')
        losses += loss.item()
        f1 += test_f1

        if i % 10 == 0:
            print(f'Epoch {i}, Train F1: {test_f1:.4f}, Loss: {loss.item():.4f}')

    return f1 / 100



def pre_train():
    memory.train()
    mgd.train()
    bin_predictor.train()
    mul_predictor.train()
    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.
    total_loss = 0
    train_gen = tqdm.tqdm(train_loader)
    for batch in train_gen:
        batch = batch.to(device)
        optimizer.zero_grad()
        src, dst, t, msg, label, attack = batch.src, batch.dst, batch.t, batch.msg, batch.label, batch.attack
        n_id = torch.cat([src, dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        z, last_update = memory(n_id)
        ed, m = nodeMap(torch.stack((src, dst), dim=0))
        ed = ed.to(device)
        norm_factor, ed = cal_norm(ed, num_nodes=len(z), self_loop=False)
        z = mgd(z, ed, norm_factor).to(device)
        # binary
        bin_out = bin_predictor(z[assoc[src]], z[assoc[dst]])
        mul_out = mul_predictor(z[assoc[src]], z[assoc[dst]])
        loss = criterion(bin_out, mul_out, label, attack, z)
        # multiple
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)
        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss) * batch.num_events
    return total_loss / data1.num_events

def meta_test():
    # memory.eval()
    # mgd.eval()
    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()
    losses = 0
    precision_score = []
    recall_score = []
    f1 = 0
    nmi = 0
    datax,_=filtered_data(data1,3)
    for i in range(10):
        support, query = sample_test(datax, 5, 20, split_ratio=0.8)
        optimizer.zero_grad()
        batch = support.to(device)
        src, dst, t, msg, label, attack = batch.src, batch.dst, batch.t, batch.msg, batch.label, batch.attack
        num_classes = torch.max(attack) + 1
        # print(src)
        n_id = torch.cat([src, dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        z, last_update = memory(n_id)
        ed, m = nodeMap(torch.stack((src, dst), dim=0))
        ed = ed.to(device)
        norm_factor, ed = cal_norm(ed, num_nodes=len(z), self_loop=False)
        z = mgd(z, ed, norm_factor).to(device)
        support_embeddings = torch.cat([z[assoc[src]], z[assoc[dst]]], 1)
        subspaces = calculate_subspaces(support_embeddings, 3, num_classes)
        # vector = model1(vector)
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)

        batch = query.to(device)
        src, dst, t, msg, label, attack = batch.src, batch.dst, batch.t, batch.msg, batch.label, batch.attack
        n_id = torch.cat([src, dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        z, last_update = memory(n_id)
        ed, m = nodeMap(torch.stack((src, dst), dim=0))
        ed = ed.to(device)
        norm_factor, ed = cal_norm(ed, num_nodes=len(z), self_loop=False)
        z = mgd(z, ed, norm_factor).to(device)
        query_embeddings = torch.cat([z[assoc[src]], z[assoc[dst]]], 1)
        
        projection_lengths = calculate_projection_lengths(query_embeddings, subspaces)
        # print("Projection lengths shape:", projection_lengths.shape)
        probabilities = calculate_probabilities(projection_lengths)
        # print("Probabilities shape:", probabilities.shape)
        memory.detach()
        
        _, preds = torch.max(probabilities, dim=1)
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)
        true = attack.cpu().numpy()
        test_f1 = f1_score(true, preds.cpu().numpy(), average='macro')

        test_nmi = normalized_mutual_info_score(true, preds.cpu().numpy())
        nmi+=test_nmi
        precision, recall, _, _ = precision_recall_fscore_support(true, preds.cpu().numpy(), average='weighted')
        precision_score.append(precision)
        recall_score.append(recall)
        memory.detach()
        f1+= test_f1
    precision_scores.append(np.mean(precision_score))
    recall_scores.append(np.mean(recall_score))
    return f1/10, nmi/10


if __name__ == '__main__':
    precision_scores = []
    recall_scores = []
    test_f1_scores = []
    test_nmi_scores = []
    for epoch in range(10): 
        loss = pre_train()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    # torch.save(memory,'./data/memory.pth')
    # torch.save(mgd,'./data/mgd.pth')
    # torch.save(assoc,'./data/memory.pth')
    f1x = 0
    for i in range(10):
        f1 = meta_train()
        print(f'train = {f1}')
        f1, nmi = meta_test()
        print(f'test F1 = {f1}, nmi = {nmi}')
        test_f1_scores.append(f1)
        test_nmi_scores.append(nmi)
    # avg_precision = np.mean(precision_scores) if precision_scores else 0
    # avg_recall = np.mean(recall_scores) if recall_scores else 0
    # print(f'Average Precision: {avg_precision:.4f} ± {std_precision:.4f}')
    # print(f'Average Recall: {avg_recall:.4f} ± {std_recall:.4f}')
    avg_f1 = np.mean(test_f1_scores)
    avg_nmi = np.mean(test_nmi_scores)
    avg_precision = np.mean(precision_scores) if precision_scores else 0
    avg_recall = np.mean(recall_scores) if recall_scores else 0
    # avg_f1 = np.mean(test_f1_scores) if test_f1_scores else 0
    # avg_nmi = np.mean(test_nmi_scores) if test_nmi_scores else 0
    std_precision = np.std(precision_scores, ddof=1) if precision_scores else 0
    std_recall = np.std(recall_scores, ddof=1) if recall_scores else 0
    std_f1 = np.std(test_f1_scores, ddof=1) if test_f1_scores else 0
    std_nmi = np.std(test_nmi_scores, ddof=1) if test_nmi_scores else 0

    # print(f'test F1 = {test}, nmi = {nmis}')
    with open('results_bot_KSCL.txt', 'w') as file:
        print(f'Average Precision: {avg_precision:.4f} ± {std_precision:.4f}', file = file)
        print(f'Average Recall: {avg_recall:.4f} ± {std_recall:.4f}', file = file)
        print(f'Average F1 Score: {avg_f1:.4f} ± {std_f1:.4f}', file = file)
        print(f'Average NMI: {avg_nmi:.4f} ± {std_nmi:.4f}', file = file)




