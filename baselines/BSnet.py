import time

import numpy as np
import random
import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from model.MGD import MGD
from model.SE import SelfExpr
from utils.LOSS import Loss
from utils.MLP import MLPPredictor
from utils.funcs import *
from sklearn.metrics import f1_score
from utils.funcs import filtered_data
from sklearn.metrics import normalized_mutual_info_score

import os

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.manifold import TSNE


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_all = torch.load("./data/CIC-ToN-IoT.pt",weights_only=False)
print(data_all)
class_pre =10
class_meta_train = 5
class_meta_test = 5
data1, data = filtered_data(data_all, class_meta_train)
data,_ = filtered_data(data_all, class_pre)
# data,_ = filtered_data(data, class_pre)
data1 = data1.to(device)
data = data.to(device)
neighbor_loader = LastNeighborLoader(data.num_nodes+data1.num_nodes, size=20, device=device)
memory_dim = time_dim = embedding_dim = 128

train_loader = TemporalDataLoader(data, batch_size=100)
layer = 1
gind_params = {'num_layers': 2, 'alpha': 0.03, 'hidden_channels': 128, 'drop_input': True, 'dropout_imp': 0.6,
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

mgd = MGD(in_channels=embedding_dim, out_channels=embedding_dim, **gind_params).to(device)

bin_predictor = MLPPredictor(in_features=embedding_dim, out_classes=2).to(device)
mul_predictor = MLPPredictor(in_features=embedding_dim, out_classes=10).to(device)

optimizer = torch.optim.Adam(set(memory.parameters()) | set(mgd.parameters()), lr=0.0001)
criterion = Loss(2, class_pre)
assoc = torch.empty(data1.num_nodes+data.num_nodes, dtype=torch.long, device=device)
semodel = SelfExpr((128)*class_meta_test).to(device)
# data_tr, data_te = filtered_data(data1, class_bi_train)
optimizer1 = torch.optim.Adam(set(semodel.parameters()))

def sample_train(datax, k, num_query, split_ratio):
    classes = [0, 1, 2, 3, 4]
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
    classes = [0, 1, 2, 3, 4]
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

# def sample1(k, num_query):
#     classes = [0, 1, 2]
#     indices_support = []
#     indices_query = []
#     datax, _ = filtered_data(data_tr, class_bi_test)

#     # print(f'attacks = {datax.attack}')
#     # print(datax)
#     for i in classes:
#         x = (np.where(datax.attack == i)[0])
#         # print(x)
#         random_selection = np.random.choice(x, size=k + num_query, replace=False)
#         selected_list = random_selection.tolist()
#         indices_support.extend(selected_list[:k])
#         indices_query.extend(selected_list[k:])
#     data_sup = datax[indices_support]
#     data_q = datax[indices_query]
#     return data_sup, data_q

def cosine_similarity(tensor_a, tensor_b):
    dot_product = torch.dot(tensor_a.flatten(), tensor_b.flatten())
    norm_a = torch.norm(tensor_a)
    norm_b = torch.norm(tensor_b)
    return dot_product / (norm_a * norm_b)

def euclidean_similarity(tensor_a, tensor_b):
    distance = torch.norm(tensor_a - tensor_b, p=2)
    similarity = -distance
    return similarity


def min_max_normalize(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data


def meta_train():
    memory.train()
    mgd.train()
    memory.reset_state()  
    neighbor_loader.reset_state()
    semodel.reset()
    losses = 0
    f1 = 0
    datax, _ = filtered_data(data1, 5)
    for i in range(100):
        support, query = sample_train(datax, 5, 20, split_ratio=0.93)
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
        for i,v in enumerate( vector):
            label_v = attack[i].item()
            if label_v not in support_dist:
                support_dist[label_v] = []
            support_dist[label_v].append(v)
        # print(support_dist)
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
        result_list = []
        result_dist = {}
        preds = []
        for v in out_vectors:
            for label , vec_list in support_dist.items():
                # print(f'label is {label}')
                result_dist[label] = 0
                for vectors in vec_list:
                    sim = cosine_similarity(v,vectors).item()
                    euc_sim = euclidean_similarity(v, vectors).item()
                    avg_sim = (sim+euc_sim)/2
                    result_dist[label] += avg_sim
                result_dist[label] = result_dist[label]/len(vec_list)
            max_key = None
            # print(result_dist)
            list_sim = []
            for key, value in result_dist.items():
                list_sim.append(value)
            # print(list_sim)
            normalized_data = min_max_normalize(list_sim)
            tensor_data = torch.tensor(normalized_data, requires_grad=True)
            softmax_output = F.softmax(tensor_data, dim=0)
            # print(softmax_output)
            result_list.append(softmax_output.tolist())
            preds.append(softmax_output.argmax(-1).item())
        result = torch.tensor(result_list, requires_grad=True).to(device)
        # pred = result.cpu().tolist()
        true = attack.cpu().tolist()
        test_f1 = f1_score(true,preds,average = 'weighted')
        loss = F.cross_entropy(result,attack)
        loss.backward()
        optimizer1.step()
        memory.detach()
        losses += loss
        f1+= test_f1
    return f1/100

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
    
    memory.reset_state()
    neighbor_loader.reset_state()
    losses = 0
    precision_score = []
    recall_score = []
    f1 = 0
    nmi = 0
    datax,_=filtered_data(data1,5)
    for i in range(10):
        support, query = sample_test(datax, 5, 20, split_ratio=0.8)
        batch = support.to(device)
        src, dst, t, msg, label, attack = batch.src, batch.dst, batch.t, batch.msg, batch.label, batch.attack
        # print(src)
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
        for i,v in enumerate( vector):
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
        result_list = []
        result_dist = {}
        preds = []
        for v in out_vectors:
            for label , vec_list in support_dist.items():
                # print(f'label is {label}')
                result_dist[label] = 0
                for vectors in vec_list:
                    sim = cosine_similarity(v,vectors).item()
                    euc_sim = euclidean_similarity(v, vectors).item()
                    avg_sim = (sim+euc_sim)/2
                    result_dist[label] += avg_sim
                result_dist[label] = result_dist[label]/len(vec_list)
            max_key = None
            # print(result_dist)
            list_sim = []
            for key, value in result_dist.items():
                list_sim.append(value)
            # print(list_sim)
            normalized_data = min_max_normalize(list_sim)
            tensor_data = torch.tensor(normalized_data, requires_grad=True)
            softmax_output = F.softmax(tensor_data, dim=0)
            # print(softmax_output)
            result_list.append(softmax_output.tolist())
            preds.append(softmax_output.argmax(-1).item())
        result = torch.tensor(result_list, requires_grad=True).to(device)
        # pred = result.cpu().tolist()
        true = attack.cpu().tolist()
        test_f1 = f1_score(true,preds,average = 'weighted')
        # print(f'test f1 = {test_f1}')
        loss = F.cross_entropy(result,attack)
        loss.backward()
        memory.detach()
        losses += loss
        f1+= test_f1
        # f1_scores.append(test_f1)
        test_nmi = normalized_mutual_info_score(true, preds)
        nmi+= test_nmi
        precision, recall, _, _ = precision_recall_fscore_support(true, preds, average='weighted')
        precision_score.append(precision)
        recall_score.append(recall)
    precision_scores.append(np.mean(precision_score))
    recall_scores.append(np.mean(recall_score))


       
    labels = true
    np.save('./picdata/labels.npy', labels)
    #t-SNE
    out_embedding = out_vectors.cpu().detach().numpy()
    # tsne = TSNE(n_components=2, perplexity=2, random_state=42, init='random', learning_rate=200.0)
    # perplexity:预估每个cluster可能有多少个元素
    tsne = TSNE(n_components=2, perplexity=20,early_exaggeration=12, random_state=42, init='random', learning_rate=1000.0)

    tsne_results = tsne.fit_transform(out_embedding)
    np.save('./picdata/t-SNE_embeddings.npy', tsne_results)
    return f1/10, nmi/10



if __name__ == '__main__':
    precision_scores = []
    recall_scores = []
    test_f1_scores = []
    test_nmi_scores = []
    # for epoch in range(50):
    #     loss = pre_train()
    #     print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
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
        plt.savefig(f'./pic/tsne{i}_BSnet.pdf', bbox_inches='tight')

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
    with open('results_ton_Bi.txt', 'w') as file:
        print(f'Average Precision: {avg_precision:.4f} ± {std_precision:.4f}', file = file)
        print(f'Average Recall: {avg_recall:.4f} ± {std_recall:.4f}', file = file)
        print(f'Average F1 Score: {avg_f1:.4f} ± {std_f1:.4f}', file = file)
        print(f'Average NMI: {avg_nmi:.4f} ± {std_nmi:.4f}', file = file)
    
    
    
    # plt.figure(figsize=(10, 5))
    # plt.plot(f1_scores, marker='o', linestyle='-', label='F1 Scores per Repetition')   
    # avg_f1_score = sum(f1_scores) / len(f1_scores)
    # plt.axhline(y=avg_f1_score, color='r', linestyle='--', label='Average F1 Score')
    # plt.title('F1 Scores per Bi-Test Repetition and Average')
    # plt.xlabel('Repetition')
    # plt.ylabel('F1 Score')
    # plt.legend()
    # plt.show()
    # plt.savefig("./pic/3D-IDS_f1_scores_and_average_bi.jpg")