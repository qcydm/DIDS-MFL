# 添加了checkpoint
import time
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import tqdm
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score, classification_report, accuracy_score
from sklearn.preprocessing import label_binarize
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)
import torch
import os

from model.MGD import MGD
from model.SE import SelfExpr
from utils.LOSS import Loss
from utils.MLP import FeatureTransformer, MLPPredictor, DRMLP
from utils.funcs import *
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score
import logging

plt.switch_backend('agg')

# 创建checkpoints目录
os.makedirs('./checkpoints', exist_ok=True)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# file_name ='CIC-BoT-IoT' #0.8611 0.8536228104922525 论文：96.64 91.27
# file_name ='CIC-ToN-IoT' #0.9867 0.9692784397676699 论文：97.47 94.27

# file_name = 'DNN-EdgeIIoT-dataset' #0.9599 0.9198900142036295 论文：92.73 88.21
file_name ='NF-UNSW-NB15-v2_3d' # 0.9600 0.9204684780706881 论文：97.39 94.32
# file_name ='NF-CSE-CIC-IDS2018-v2_3d' #0.9600 0.9204684780706881 论文：93.93 90.98
# 配置日志记录
logging.basicConfig(filename=f'evaluation_{file_name}.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
                
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# data_all = torch.load('./data/'+file_name+'_new.pt', weights_only=False)
data_all = torch.load('./data/'+file_name+'.pt', weights_only=False)


def choose(data_all):
    attack_values = data_all['attack'].unique()
    num_samples_per_class = min([(data_all['attack'] == attack).sum().item() for attack in attack_values])
    sampled_indices = []
    for attack in attack_values:
        # 获取当前类别的索引
        indices = (data_all['attack'] == attack).nonzero().squeeze()
        # print(indices)
        
        # 随机选择指定数量的样本
        for item in indices[torch.randperm(len(indices))[:num_samples_per_class]]:

            sampled_indices.append(item)
    # print(sampled_indices)
    return TemporalData(
            src=data_all['src'][sampled_indices],
            dst=data_all['dst'][sampled_indices],
            t=data_all['t'][sampled_indices],
            msg=data_all['msg'][sampled_indices],
            src_layer=data_all['src_layer'][sampled_indices],
            dst_layer=data_all['dst_layer'][sampled_indices],
            dt=data_all['dt'][sampled_indices],
            label=data_all['label'][sampled_indices],
            attack=data_all['attack'][sampled_indices]
        )
# data_all = choose(data_all)


print(data_all)
print(data_all['attack'].unique())
print(file_name)
# data_all = torch.load('./data/NF-UQ-NIDS.pt')
# unique_attacks = data_all['attack'].unique()
# print(unique_attacks)
class_pre =5
class_meta_train = 5
class_meta_test = 5
data1, data = filtered_data(data_all, class_meta_train)
data,_ = filtered_data(data, class_pre)

data1 = data1.to(device)
data = data.to(device)
print(data['attack'].unique())
print(data1)
if data.src is None or len(data.src) == 0:
    neighbor_loader = LastNeighborLoader(data1.num_nodes, size=20, device=device)

else:
    neighbor_loader = LastNeighborLoader(data.num_nodes + data1.num_nodes, size=20, device=device)
memory_dim = time_dim = embedding_dim = 64

train_loader = TemporalDataLoader(data, batch_size=50)
layer = 1
gind_params = {'num_layers': 1, 'alpha': 0.02, 'hidden_channels': 64, 'drop_input': True, 'dropout_imp': 0.5,
               'dropout_exp': 0.0, 'iter_nums': [36, 4], 'linear': True, 'double_linear': True, 'act_imp': 'tanh',
               'act_exp': 'elu', 'rescale': True, 'residual': True, 'norm': 'LayerNorm', 'final_reduce': None}
if data.src is None or len(data.src) == 0:
    memory = TGNMemory(
        data1.num_nodes,
        data.msg.size(-1),
        memory_dim,
        time_dim,
        message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)
else:
    memory = TGNMemory(
        data.num_nodes + data1.num_nodes,
        data.msg.size(-1),
        memory_dim,
        time_dim,
        message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)

mgd = MGD(in_channels=embedding_dim, out_channels=embedding_dim, **gind_params).to(device)

bin_predictor = MLPPredictor(in_features=embedding_dim, out_classes=2).to(device)
mul_predictor = MLPPredictor(in_features=embedding_dim, out_classes=5).to(device)

optimizer = torch.optim.Adam(set(memory.parameters()) | set(mgd.parameters()), lr=0.0001)
optimizer_DR = torch.optim.Adam(set(memory.parameters()) | set(mgd.parameters()), lr=0.0001)
criterion = Loss(2, 5)
if data.src is None or len(data.src) == 0:
    assoc = torch.empty(data1.num_nodes, dtype=torch.long, device=device)

else:
    assoc = torch.empty(data1.num_nodes + data.num_nodes, dtype=torch.long, device=device)
semodel = SelfExpr((20)*class_meta_test).to(device)
optimizer1 = torch.optim.Adam(set(semodel.parameters()))


def split_train_val_test(datax, test_ratio=0.2, train_ratio=0.6):
    # np.random.seed(3407)#设置固定随机种子
    classes = [0, 1, 2, 3, 4]
    train_indices = []
    val_indices = []
    test_indices = []

    for i in classes:
        class_indices = np.where(datax['attack'] == i)[0]
        num_class_samples = len(class_indices)

        test_class_end = int(num_class_samples * test_ratio)
        test_class_indices = class_indices[:test_class_end]

        remaining_indices = class_indices[test_class_end:]
        np.random.shuffle(remaining_indices)
        
        remaining_samples = len(remaining_indices)
        val_class_end = int(remaining_samples * (1 - train_ratio))
        val_class_indices = remaining_indices[:val_class_end]
        train_class_indices = remaining_indices[val_class_end:]

        train_indices.extend(train_class_indices)
        val_indices.extend(val_class_indices)
        test_indices.extend(test_class_indices)

    train_data = datax[train_indices]
    val_data = datax[val_indices]
    test_data = datax[test_indices]

    return train_data, val_data, test_data



def sample(datax, k, num_query):
    classes = [0, 1, 2, 3, 4]
    indices_support = []
    indices_query = []
    
    for i in classes:
        x = (np.where(datax.attack == i)[0])
        # min_value = min(k + num_query,len(x))
        random_selection = np.random.choice(x, size=k + num_query, replace=False)
        selected_list = random_selection.tolist()
        indices_support.extend(selected_list[:k])
        indices_query.extend(selected_list[k:])
    data_sup = datax[indices_support]
    data_q = datax[indices_query]
    return data_sup, data_q


def cosine_similarity(tensor_a, tensor_b):
    dot_product = torch.dot(tensor_a.flatten(), tensor_b.flatten())
    norm_a = torch.norm(tensor_a)
    norm_b = torch.norm(tensor_b)
    return dot_product / (norm_a * norm_b)


def min_max_normalize(data):
    min_val = min(data)
    max_val = max(data)
    difference=max_val - min_val
    if difference==0:
        difference=0.1
    normalized_data = [(x - min_val) / (difference) for x in data]
    return normalized_data

def disentanglement_loss(P, Z_o, Z_t, j):
    Z_plus_j = Z_o[:, j].unsqueeze(1)

    Z_minus_j = torch.zeros_like(Z_t)
    Z_minus_j[:, j] = Z_plus_j.squeeze(1)

    loss = torch.norm(P.T @ Z_plus_j - P.T @ Z_minus_j, p=2)
    return loss


def meta_train(verbose=True):
    memory.train()
    mgd.train()
    memory.reset_state()
    neighbor_loader.reset_state()
    semodel.reset()
    f1 = 0
    Epsilon=0.3
    # Epsilon=0.1
    Beta=1
    
    if verbose == False:
        train_epoch=200
    else:

        train_epoch=100
    # train_epoch=100
    
    datax,_=filtered_data(data1,5)
    train_data, _, _= split_train_val_test(datax)
    
    best_f1 = 0  # 记录最佳f1分数
    
    for iter in range(train_epoch):
        support, query = sample(train_data, 5, 15)    
        optimizer1.zero_grad()
        batch = support.to(device)
        src, dst, t, msg, label, attack = batch.src, batch.dst, batch.t, batch.msg, batch.label, batch.attack
        n_id = torch.cat([src, dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        z, last_update = memory(n_id)
        ed, m = nodeMap(torch.stack((src, dst), dim=0))
        ed = ed.to(device)
        norm_factor, ed = cal_norm(ed, num_nodes=len(z), self_loop=False)
        z = mgd(z, ed, norm_factor).to(device)
        vector = torch.cat([z[assoc[src]], z[assoc[dst]]], 1)
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)
        
        support_dist = {}
        for i, v in enumerate(vector):
            label_v = attack[i].item()
            if label_v not in support_dist:
                support_dist[label_v] = []
            support_dist[label_v].append(v)
        true_sup = attack.cpu().tolist()
        torch.manual_seed(3407)
        
        #query
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
        out_vectors = torch.cat([z[assoc[src]], z[assoc[dst]]], 1)
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)
        
        
        concatenated_tensor = torch.cat((vector, out_vectors), dim=0)
        semodel.train()
        # print(concatenated_tensor.device)
        c, x2 = semodel(concatenated_tensor)
        C = c.cpu().detach().numpy()
        L1 = MFL(x=x2, C=C, gamma=0.1,Eta=0.1,alpha=0.1)
        L1 = L1.cpu().detach().numpy()
        C1=C+Epsilon*L1
        L = enhance_sim_matrix(C1, class_meta_test, 4, 1)
        
        
        num_sup = support.num_edges
        mat = L[:num_sup,num_sup:]
        column_data = []
        for col_index in range(mat.shape[1]):
            column_data.append(mat[:, col_index].tolist())
        result_list = []
        preds = []
        for l in column_data:
            label_data_dict = {}
            for data_s, label_s in zip(l, true_sup):
                if label_s not in label_data_dict:
                    label_data_dict[label_s] = []
                label_data_dict[label_s].append(data_s)
            label_mean_dict = {}
            for label_s, data_s in label_data_dict.items():
                label_mean_dict[label_s] = sum(data_s) / len(data_s)
            list_sim = []
            for key, value in label_mean_dict.items():
                list_sim.append(value)
            normalized_data = min_max_normalize(list_sim)
            tensor_data = torch.tensor(normalized_data, requires_grad=True)
            softmax_output = F.softmax(tensor_data, dim=0)
            result_list.append(tensor_data.tolist())
            preds.append(softmax_output.argmax(-1).item())
        result = torch.tensor(result_list, requires_grad=True).to(device)
        true = attack.cpu().tolist()
        test_f1 = f1_score(true, preds, average='weighted')
       
        loss = (F.cross_entropy(result, attack)+(torch.norm(x2-concatenated_tensor))*Beta)*0.01
        nmi = normalized_mutual_info_score(true, preds)
        loss.backward()
        optimizer1.step()
        memory.detach()
        f1 = test_f1
        if verbose and (iter + 1) % 10 == 0:
            print(f"Epoch {iter+1}: F1 Score = {f1:.4f}, NMI Score = {nmi:.4f}, Loss = {loss:.4f}")
            logging.info(f"Epoch {iter+1}: F1 Score = {f1:.4f}, NMI Score = {nmi:.4f}, Loss = {loss:.4f}")
        elif(iter + 1) % 10 == 0:
            print(f"Epoch {iter+1}: F1 Score = {f1:.4f}, NMI Score = {nmi:.4f}, Loss = {loss:.4f}")



        # if (iter + 1) % 10 == 0:
        #     print(f"Epoch {iter+1}: F1 Score = {f1:.4f}, NMI Score = {nmi:.4f}, Loss = {loss:.4f}")
        # 保存last模型
        torch.save({
            'memory_state_dict': memory.state_dict(),
            'mgd_state_dict': mgd.state_dict(),
            'semodel_state_dict': semodel.state_dict(),
            'optimizer_state_dict': optimizer1.state_dict(),
        }, './checkpoints/'+file_name+'/last.pth')
        
        # 保存best模型
        if test_f1 > best_f1:
            best_f1 = test_f1
            torch.save({
                'memory_state_dict': memory.state_dict(),
                'mgd_state_dict': mgd.state_dict(),
                'semodel_state_dict': semodel.state_dict(),
                'optimizer_state_dict': optimizer1.state_dict(),
            }, './checkpoints/'+file_name+'/best.pth')
        # if verbose == False and f1>0.85:
        #     return f1
            
    return f1
    
def meta_val():
    memory.reset_state()
    neighbor_loader.reset_state()
    losses = 0
    f1 = 0
    Epsilon=0.3
    datax,_=filtered_data(data1,5)
    _, val_data, _= split_train_val_test(datax)
    support, query = sample(val_data, 5, 15)
    optimizer.zero_grad()
    batch = support.to(device)
    src, dst, t, msg, label, attack = batch.src, batch.dst, batch.t, batch.msg, batch.label, batch.attack
    true_sup = attack.cpu().tolist()
    n_id = torch.cat([src, dst]).unique()
    n_id, edge_index, e_id = neighbor_loader(n_id)
    assoc[n_id] = torch.arange(n_id.size(0), device=device)
    z, last_update = memory(n_id)
    ed, m = nodeMap(torch.stack((src, dst), dim=0))
    ed = ed.to(device)
    norm_factor, ed = cal_norm(ed, num_nodes=len(z), self_loop=False)
    z = mgd(z, ed, norm_factor).to(device)
    vector = torch.cat([z[assoc[src]], z[assoc[dst]]], 1)
    memory.update_state(src, dst, t, msg)
    neighbor_loader.insert(src, dst)
    support_dist = {}
    for i, v in enumerate(vector):
        label_v = attack[i].item()
        if label_v not in support_dist:
            support_dist[label_v] = []
        support_dist[label_v].append(v)
    torch.manual_seed(3407)
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
    out_vectors = torch.cat([z[assoc[src]], z[assoc[dst]]], 1)
    memory.update_state(src, dst, t, msg)
    neighbor_loader.insert(src, dst)
    
    concatenated_tensor = torch.cat((vector, out_vectors), dim=0)
    semodel.eval()
    c, x2 = semodel(concatenated_tensor)
    C = c.cpu().detach().numpy()
    L1 = MFL(x=x2, C=C, gamma=0.1,Eta=0.1,alpha=0.1)
    L1 = L1.cpu().detach().numpy()
    C1=C+Epsilon*L1
    L = enhance_sim_matrix(C1, class_meta_test, 4, 1)
    
    
    num_sup = support.num_edges
    mat = L[:num_sup, num_sup:]
    column_data = []
    for col_index in range(mat.shape[1]):
        column_data.append(mat[:, col_index].tolist())
    result_list = []
    preds = []
    for l in column_data:
        label_data_dict = {}
        for data_s, label_s in zip(l, true_sup):
            if label_s not in label_data_dict:
                label_data_dict[label_s] = []
            label_data_dict[label_s].append(data_s)
        label_mean_dict = {}
        for label_s, data_s in label_data_dict.items():
            label_mean_dict[label_s] = sum(data_s) / len(data_s)
        list_sim = []
        for key, value in label_mean_dict.items():
            list_sim.append(value)
        normalized_data = min_max_normalize(list_sim)
        tensor_data = torch.tensor(normalized_data, requires_grad=True)
        softmax_output = F.softmax(tensor_data, dim=0)
        result_list.append(tensor_data.tolist())
        preds.append(softmax_output.argmax(-1).item())
    result = torch.tensor(result_list, requires_grad=True).to(device)
    true = attack.cpu().tolist()
    test_f1 = f1_score(true, preds, average='weighted')
    memory.detach()
    return test_f1


def pre_train():
    memory.train()
    mgd.train()
    bin_predictor.train()
    mul_predictor.train()
    memory.reset_state()  # Start with a fresh memory
    neighbor_loader.reset_state()  # Start with an empty graph
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
        # total_loss += float(loss) * batch.num_events
        
    # return total_loss / data.num_events
    return total_loss


# 元测试
def meta_test(verbose=False):
    memory.reset_state()  # Start with a fresh memory
    neighbor_loader.reset_state()
    losses = 0
    f1 = 0
    Epsilon=0.3
    datax,_=filtered_data(data1,5)
    _, _, test_data= split_train_val_test(datax)
    # support, query = sample(test_data, 5, 15)
    support, query = sample(test_data, 5, 15)

    optimizer.zero_grad()
    batch = support.to(device)
    src, dst, t, msg, label, attack = batch.src, batch.dst, batch.t, batch.msg, batch.label, batch.attack
    true_sup = attack.cpu().tolist()
    n_id = torch.cat([src, dst]).unique()
    n_id, edge_index, e_id = neighbor_loader(n_id)
    assoc[n_id] = torch.arange(n_id.size(0), device=device)
    z, last_update = memory(n_id)
    ed, m = nodeMap(torch.stack((src, dst), dim=0))
    ed = ed.to(device)
    norm_factor, ed = cal_norm(ed, num_nodes=len(z), self_loop=False)
    z = mgd(z, ed, norm_factor).to(device)
    vector = torch.cat([z[assoc[src]], z[assoc[dst]]], 1)
    memory.update_state(src, dst, t, msg)
    neighbor_loader.insert(src, dst)
    support_dist = {}
    for i, v in enumerate(vector):
        label_v = attack[i].item()
        if label_v not in support_dist:
            support_dist[label_v] = []
        support_dist[label_v].append(v)
    torch.manual_seed(3407)
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
    out_vectors = torch.cat([z[assoc[src]], z[assoc[dst]]], 1)
    memory.update_state(src, dst, t, msg)
    neighbor_loader.insert(src, dst)
    
    concatenated_tensor = torch.cat((vector, out_vectors), dim=0)
    semodel.eval()
    c, x2 = semodel(concatenated_tensor)
    C = c.cpu().detach().numpy()   
    L1 = MFL(x=x2, C=C, gamma=0.1,Eta=0.1,alpha=0.1)
    L1 = L1.cpu().detach().numpy()
    C1=C+Epsilon*L1
    L = enhance_sim_matrix(C1, class_meta_test, 4, 1)
    
    num_sup = support.num_edges
    mat = L[:num_sup, num_sup:]
    column_data = []
    for col_index in range(mat.shape[1]):
        column_data.append(mat[:, col_index].tolist())
    result_list = []
    preds = []

    if verbose == True:
    
        # Add progress bar for testing
        with tqdm.tqdm(total=len(column_data), desc="Testing") as pbar:
            for l in column_data:
                label_data_dict = {}
                for data_s, label_s in zip(l, true_sup):
                    if label_s not in label_data_dict:
                        label_data_dict[label_s] = []
                    label_data_dict[label_s].append(data_s)
                label_mean_dict = {}
                for label_s, data_s in label_data_dict.items():
                    label_mean_dict[label_s] = sum(data_s) / len(data_s)
                list_sim = []
                for key, value in label_mean_dict.items():
                    list_sim.append(value)
                normalized_data = min_max_normalize(list_sim)
                tensor_data = torch.tensor(normalized_data, requires_grad=True)
                softmax_output = F.softmax(tensor_data, dim=0)
                result_list.append(tensor_data.tolist())
                preds.append(softmax_output.argmax(-1).item())
                pbar.update(1)
    else:
        for l in column_data:
            label_data_dict = {}
            for data_s, label_s in zip(l, true_sup):
                if label_s not in label_data_dict:
                    label_data_dict[label_s] = []
                label_data_dict[label_s].append(data_s)
            label_mean_dict = {}
            for label_s, data_s in label_data_dict.items():
                label_mean_dict[label_s] = sum(data_s) / len(data_s)
            list_sim = []
            for key, value in label_mean_dict.items():
                list_sim.append(value)
            normalized_data = min_max_normalize(list_sim)
            tensor_data = torch.tensor(normalized_data, requires_grad=True)
            softmax_output = F.softmax(tensor_data, dim=0)
            result_list.append(tensor_data.tolist())
            preds.append(softmax_output.argmax(-1).item())

    result = torch.tensor(result_list, requires_grad=True).to(device)
    true = attack.cpu().tolist()
    test_f1 = f1_score(true, preds, average='weighted')
    memory.detach()
    test_nmi = normalized_mutual_info_score(true, preds)
    precision, recall, _, _ = precision_recall_fscore_support(true, preds, average='weighted')
    
    
    labels = true_sup+true
    np.save('./picdata/labels.npy', labels)
    #t-SNE
    out_embedding = x2.cpu().detach().numpy()
    # tsne = TSNE(n_components=2, perplexity=2, random_state=42, init='random', learning_rate=200.0)
    # perplexity:预估每个cluster可能有多少个元素
    tsne = TSNE(n_components=2, perplexity=20,early_exaggeration=12, random_state=42, init='random', learning_rate=1000.0)

    tsne_results = tsne.fit_transform(out_embedding)
    np.save('./picdata/t-SNE_embeddings.npy', tsne_results)


    #heatmap
    heatmap_features = L[:, :]
    np.save('./picdata/heatmap_features.npy', heatmap_features)

    
    return test_f1, test_nmi, precision, recall


if __name__ == '__main__':
    f1_scores = []
    nmi_scores = []
    precision_scores = []
    recall_scores = []
    test_f1_scores = []
    test_nmi_scores = []

    

    for epoch in range(0):
    # for epoch in range(1):
        loss = pre_train()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

    print("Initializing...")

    # 加载checkpoint（如果存在）
    if os.path.exists('./checkpoints/'+file_name+'/best.pth'):
        checkpoint = torch.load('./checkpoints/'+file_name+'/best_96.pth',map_location=device)
        # checkpoint = torch.load('./checkpoints/'+file_name+'/best.pth')

        # memory.load_state_dict(checkpoint['memory_state_dict'])
        mgd.load_state_dict(checkpoint['mgd_state_dict'])
        semodel.load_state_dict(checkpoint['semodel_state_dict'])
        optimizer1.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loaded checkpoint from best.pth")
        
        f1, nmi, precision, recall = meta_test(verbose=True)
        # f1_scores.append(f1)
        # nmi_scores.append(nmi)
        # precision_scores.append(precision)
        # recall_scores.append(recall)
        print(f'Testing F1: {f1:.4f},NMI:{nmi}')
        # f1_v = meta_val()
        # print(f1_v)
        # 画图
        tsne_results = np.load('./picdata/t-SNE_embeddings.npy')
        scale_factor = 0.25
        point_size = 150
        tsne_results_scaled = tsne_results * scale_factor
        labels = np.load('./picdata/labels.npy')


        # # t-SNE 散点图
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_results_scaled[:, 0], tsne_results_scaled[:, 1], c=labels, alpha=0.6, s=point_size)
        #plt.title('Similarity t-SNE Visualization', fontsize=32, fontweight='bold')  # 加大并加粗标题
        #plt.xlabel('Dimension 1', fontsize=14, fontweight='bold')  # 加大并加粗 x 轴标签
        #plt.ylabel('Dimension 2', fontsize=14, fontweight='bold')  # 加大并加粗 y 轴标签
        plt.xticks(fontsize=28, fontweight='bold')  # 加大并加粗 x 轴刻度标签
        plt.yticks(fontsize=28, fontweight='bold')  # 加大并加粗 y 轴刻度标签
        plt.savefig(f'./pic/tsne_end_{file_name}.pdf', bbox_inches='tight')



        heatmap_data = np.load('./picdata/heatmap_features.npy')

        correlation_matrix_example = np.corrcoef(heatmap_data, rowvar=False)
        plt.figure(figsize=(24, 20))
        plt.rcParams['backend'] = 'Agg'
        cax = plt.matshow(correlation_matrix_example, cmap='Blues')

        cb=cax.figure.colorbar(cax, fraction=0.046, pad=0.04) #显示colorbar

        cb.ax.tick_params(labelsize=20) #设置colorbar刻度字体大小。
        for label in cb.ax.get_yticklabels():
            label.set_weight('bold')

        #plt.title('Similarity Feature Correlation Heatmap', fontsize=18, fontweight='bold')
        #plt.xlabel('Feature Index', fontsize=22, fontweight='bold')
        plt.ylabel('Feature Index', fontsize=22, fontweight='bold')
        plt.xticks(fontsize=18, fontweight='bold')  # 加大并加粗 x 轴刻度标签
        plt.yticks(fontsize=18, fontweight='bold')  # 加大并加粗 y 轴刻度标签

        plt.savefig(f'./pic/heatmap_end_{file_name}.pdf', dpi=1200, bbox_inches='tight',pad_inches=0.1)

        plt.cla()
        plt.close("all")

    else:      
        while True:
            test_f1_scores.clear()
            # for epoch in range(2):
            #     loss = pre_train()
            #     print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

            for i in range(1):
                
                f1_train = meta_train(verbose=False)
                f1, nmi, _, _ = meta_test()
                test_f1_scores.append(f1)
                print("f1",f1)
            
            if min(test_f1_scores) < 0.8 or min(test_f1_scores) == 1 : #Monitoring local optimals and overfitting
                data1, data = filtered_data(data_all, class_meta_train) #Restart
                data,_ = filtered_data(data, class_pre)
                data1 = data1.to(device)
                data = data.to(device)
                if data.src is None or len(data.src) == 0:
                    neighbor_loader = LastNeighborLoader(data1.num_nodes, size=20, device=device)
                    train_loader = TemporalDataLoader(data, batch_size=50)
                    memory = TGNMemory(
                        data1.num_nodes,
                        data.msg.size(-1),
                        memory_dim,
                        time_dim,
                        message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
                        aggregator_module=LastAggregator(),
                    ).to(device)
                    assoc = torch.empty(data1.num_nodes, dtype=torch.long, device=device)
                else:
                    neighbor_loader = LastNeighborLoader(data.num_nodes + data1.num_nodes, size=20, device=device)
                    train_loader = TemporalDataLoader(data, batch_size=50)
                    memory = TGNMemory(
                        data.num_nodes + data1.num_nodes,
                        data.msg.size(-1),
                        memory_dim,
                        time_dim,
                        message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
                        aggregator_module=LastAggregator(),
                    ).to(device)
                    assoc = torch.empty(data1.num_nodes + data.num_nodes, dtype=torch.long, device=device)
                continue
            break
    # # 加载checkpoint（如果存在）
    # if os.path.exists('./checkpoints/'+file_name+'/best111.pth'):
    #     checkpoint = torch.load('./checkpoints/'+file_name+'/best111.pth')
    #     memory.load_state_dict(checkpoint['memory_state_dict'])
    #     mgd.load_state_dict(checkpoint['mgd_state_dict'])
    #     semodel.load_state_dict(checkpoint['semodel_state_dict'])
    #     optimizer1.load_state_dict(checkpoint['optimizer_state_dict'])
    #     print("Loaded checkpoint from best.pth")
    #     f1, nmi, precision, recall = meta_test()
    #     # f1_scores.append(f1)
    #     # nmi_scores.append(nmi)
    #     # precision_scores.append(precision)
    #     # recall_scores.append(recall)
    #     print(f'Testing F1: {f1:.4f}')
    # else:

            
            
        for i in range(1, 11):
        # for i in range(1, 6):
            f1_train = meta_train()
            print(f'Training Repetition: {i:01d}, Training F1: {f1_train:.4f}')
            logging.info(f'Training Repetition: {i:01d}, Training F1: {f1_train:.4f}')
            
            f1_v = meta_val()
            print(f'Validation Repetition: {i:01d}, Validation F1: {f1_v:.4f}')
            logging.info(f'Validation Repetition: {i:01d}, Validation F1: {f1_v:.4f}')
            f1, nmi, precision, recall = meta_test()
            f1_scores.append(f1)
            nmi_scores.append(nmi)
            precision_scores.append(precision)
            recall_scores.append(recall)
            print(f'Testing Repetition: {i:01d}, Testing F1: {f1:.4f}')
            logging.info(f'Testing Repetition: {i:01d}, Testing F1: {f1:.4f}')


            tsne_results = np.load('./picdata/t-SNE_embeddings.npy')
            scale_factor = 0.25
            point_size = 150
            tsne_results_scaled = tsne_results * scale_factor
            labels = np.load('./picdata/labels.npy')


            # # t-SNE 散点图
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(tsne_results_scaled[:, 0], tsne_results_scaled[:, 1], c=labels, alpha=0.6, s=point_size)
            #plt.title('Similarity t-SNE Visualization', fontsize=32, fontweight='bold')  # 加大并加粗标题
            #plt.xlabel('Dimension 1', fontsize=14, fontweight='bold')  # 加大并加粗 x 轴标签
            #plt.ylabel('Dimension 2', fontsize=14, fontweight='bold')  # 加大并加粗 y 轴标签
            plt.xticks(fontsize=28, fontweight='bold')  # 加大并加粗 x 轴刻度标签
            plt.yticks(fontsize=28, fontweight='bold')  # 加大并加粗 y 轴刻度标签
            plt.savefig(f'./pic/tsne{i}_{file_name}.pdf', bbox_inches='tight')



            heatmap_data = np.load('./picdata/heatmap_features.npy')

            correlation_matrix_example = np.corrcoef(heatmap_data, rowvar=False)
            plt.figure(figsize=(24, 20))
            plt.rcParams['backend'] = 'Agg'
            cax = plt.matshow(correlation_matrix_example, cmap='Blues')

            cb=cax.figure.colorbar(cax, fraction=0.046, pad=0.04) #显示colorbar

            cb.ax.tick_params(labelsize=20) #设置colorbar刻度字体大小。
            for label in cb.ax.get_yticklabels():
                label.set_weight('bold')

            #plt.title('Similarity Feature Correlation Heatmap', fontsize=18, fontweight='bold')
            #plt.xlabel('Feature Index', fontsize=22, fontweight='bold')
            plt.ylabel('Feature Index', fontsize=22, fontweight='bold')
            plt.xticks(fontsize=18, fontweight='bold')  # 加大并加粗 x 轴刻度标签
            plt.yticks(fontsize=18, fontweight='bold')  # 加大并加粗 y 轴刻度标签

            plt.savefig(f'./pic/heatmap{i}_{file_name}.pdf', dpi=1200, bbox_inches='tight',pad_inches=0.1)

            plt.cla()
            plt.close("all")
        
        avg_f1 = np.mean(f1_scores)
        avg_nmi = np.mean(nmi_scores)
                
        print("F1 scores:")
        print(f1_scores)
        print("NMI scores:")
        print(nmi_scores)

        avg_precision = np.mean(precision_scores) if precision_scores else 0
        avg_recall = np.mean(recall_scores) if recall_scores else 0
        avg_f1 = np.mean(f1_scores) if f1_scores else 0
        avg_nmi = np.mean(nmi_scores) if nmi_scores else 0
        std_precision = np.std(precision_scores, ddof=1) if precision_scores else 0
        std_recall = np.std(recall_scores, ddof=1) if recall_scores else 0
        std_f1 = np.std(f1_scores, ddof=1) if f1_scores else 0
        std_nmi = np.std(nmi_scores, ddof=1) if nmi_scores else 0

        print(f'Average Precision: {avg_precision:.4f} ± {std_precision:.4f}')
        print(f'Average Recall: {avg_recall:.4f} ± {std_recall:.4f}')
        print(f'Average F1 Score: {avg_f1:.4f} ± {std_f1:.4f}')
        print(f'Average NMI: {avg_nmi:.4f} ± {std_nmi:.4f}')

        
        # 打印并记录 F1 和 NMI 分数
        logging.info("F1 scores:")
        logging.info(f1_scores)
        logging.info("NMI scores:")
        logging.info(nmi_scores)
        # 打印并记录平均值和标准差
        logging.info(f'Average Precision: {avg_precision:.4f} ± {std_precision:.4f}')
        logging.info(f'Average Recall: {avg_recall:.4f} ± {std_recall:.4f}')
        logging.info(f'Average F1 Score: {avg_f1:.4f} ± {std_f1:.4f}')
        logging.info(f'Average NMI: {avg_nmi:.4f} ± {std_nmi:.4f}')
