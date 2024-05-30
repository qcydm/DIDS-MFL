import time

import numpy as np
import random
from sklearn.manifold import TSNE
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
import torch.nn as nn
from model.MGD import MGD
from utils.LOSS import Loss
from utils.MLP import MLPPredictor
from utils.funcs import *
from sklearn.metrics import f1_score
from utils.funcs import filtered_data
from sklearn.metrics import normalized_mutual_info_score

import os

from utils.misc import Averager, count_acc
from matplotlib import pyplot as plt
plt.switch_backend('agg')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_all = torch.load("../data/CIC-BoT-IoT.pt")
class_pre =5
class_meta_train = 5
class_meta_test = 5
data1,data = filtered_data(data_all,class_meta_train)
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
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
class TrainerConfig:
    def __init__(self, model_type='ResNet', dataset='MiniImageNet', phase='meta_train', seed=0, gpu='1', dataset_dir='./dataloader/data/fc100',
                 max_epoch=100, num_batch=100, shot=1, way=5, train_query=15, val_query=15, meta_lr1=0.0001, meta_lr2=0.001, base_lr=0.01, update_step=50,
                 step_size=10, gamma=0.5, init_weights=None, eval_weights=None, meta_label='exp1',
                 pre_max_epoch=100, pre_batch_size=128, pre_lr=0.1, pre_gamma=0.2, pre_step_size=30, pre_custom_momentum=0.9, pre_custom_weight_decay=0.0005):
        self.model_type = model_type
        self.dataset = dataset
        self.phase = phase
        self.seed = seed
        self.gpu = gpu
        self.dataset_dir = dataset_dir
        self.max_epoch = max_epoch
        self.num_batch = num_batch
        self.shot = shot
        self.way = way
        self.train_query = train_query
        self.val_query = val_query
        self.meta_lr1 = meta_lr1
        self.meta_lr2 = meta_lr2
        self.base_lr = base_lr
        self.update_step = update_step
        self.step_size = step_size
        self.gamma = gamma
        self.init_weights = init_weights
        self.eval_weights = eval_weights
        self.meta_label = meta_label
        self.pre_max_epoch = pre_max_epoch
        self.pre_batch_size = pre_batch_size
        self.pre_lr = pre_lr
        self.pre_gamma = pre_gamma
        self.pre_step_size = pre_step_size
        self.pre_custom_momentum = pre_custom_momentum
        self.pre_custom_weight_decay = pre_custom_weight_decay
        
args = TrainerConfig(
    model_type='ResNet',
    dataset='MiniImageNet',
    phase='meta_train',
    seed=0,
    gpu='1',
    dataset_dir='./dataloader/data/classified_data',
    max_epoch=10,
    num_batch=100,
    shot=5,
    way=5,
    train_query=15,
    val_query=15,
    meta_lr1=0.0001,
    meta_lr2=0.0003,
    base_lr=0.003,
    update_step=50,
    step_size=10,
    gamma=0.5,
    init_weights=None,
    eval_weights=None,
    meta_label='exp1',
    pre_max_epoch=50,
    pre_batch_size=50,
    pre_lr=0.1,
    pre_gamma=0.2,
    pre_step_size=30,
    pre_custom_momentum=0.9,
    pre_custom_weight_decay=0.0005
)
# #lr2=0.001,base_lr=0.01

mgd = MGD(in_channels=embedding_dim, out_channels=embedding_dim, **gind_params).to(device)

bin_predictor = MLPPredictor(in_features=embedding_dim, out_classes=2).to(device)
mul_predictor = MLPPredictor(in_features=embedding_dim, out_classes=5).to(device)

        
class BaseLearner(nn.Module):
    """The class for inner loop."""

    def __init__(self, args, z_dim):
        super().__init__()
        self.args = args
        self.z_dim = z_dim
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([self.args.way, self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)
        self.fc1_b = nn.Parameter(torch.zeros(self.args.way))
        self.vars.append(self.fc1_b)
        # self.W = torch.nn.Linear(in_features * 2, out_classes)

    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        fc1_b = the_vars[1]
        net = F.linear(input_x, fc1_w, fc1_b)
        return net

    def parameters(self):
        return self.vars
    
# optimizer = torch.optim.Adam(set(memory1.parameters()) | set(mgd.parameters()) | set(memory1.parameters()), lr=0.0001)
optimizer = torch.optim.Adam(set(memory.parameters()) | set(mgd.parameters()), lr=0.0001)
criterion = Loss(2, 5)
assoc = torch.empty(data1.num_nodes+data.num_nodes, dtype=torch.long, device=device)
base_learner = BaseLearner(args, 128).to(device)
optimizer_meta = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, mgd.parameters())},{'params': base_learner.parameters(), 'lr': args.meta_lr2}], lr=args.meta_lr1)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_meta, step_size=args.pre_step_size,gamma=args.pre_gamma)



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

# def sample(datax,k, num_query):
#     classes = [0,1,2]
#     indices_support = []
#     indices_query = []
    
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
# support, query = sample(5, 15)

def cosine_similarity(tensor_a, tensor_b):
    dot_product = torch.dot(tensor_a.flatten(), tensor_b.flatten())
    norm_a = torch.norm(tensor_a)
    norm_b = torch.norm(tensor_b)
    return dot_product / (norm_a * norm_b)


def min_max_normalize(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data

def meta_train():
    memory.train()
    mgd.train()
    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()
    losses = 0
    f1 = 0
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['max_acc_epoch'] = 0
    datax,_ = filtered_data(data1, 5)
    for epoch in range(1, args.max_epoch+1):
        support, query = sample_train(datax, 5, 20, split_ratio=0.8)
        optimizer_meta.zero_grad()
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

        predicts = base_learner(out_vectors)
        loss = F.cross_entropy(predicts, attack)
        grad = torch.autograd.grad(loss, base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - args.base_lr * p[0], zip(grad, base_learner.parameters())))

        predicts_q = base_learner(vector, fast_weights)

        for _ in range(1, args.update_step):
            predicts = base_learner(out_vectors, fast_weights)
            loss = F.cross_entropy(predicts, attack)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - args.base_lr * p[0], zip(grad, fast_weights)))
            predicts_q = base_learner(out_vectors, fast_weights)
        loss = F.cross_entropy(predicts_q, attack)
        torch.autograd.set_detect_anomaly = True
        # optimizer_meta.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_meta.step()

        preds = predicts_q.argmax(dim=1).cpu().tolist()
        true = attack.cpu().tolist()
        f1_sco = f1_score(true, preds, average = 'weighted')
        f1+=f1_sco

        lr_scheduler.step()
        # Update the averagers
        torch.cuda.empty_cache()
    return f1/args.max_epoch

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
    base_learner.eval()
    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()
    losses = 0
    f1 = 0
    nmi = 0
    precision_score = []
    recall_score = []
    datax,_ = filtered_data(data1, 5)
    for i in range(10):
        support, query = sample_test(datax, 5, 20, split_ratio=0.8)
        optimizer.zero_grad()
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
        for i, v in enumerate(vector):
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

        predicts = base_learner(out_vectors)
        loss = F.cross_entropy(predicts, attack)
        grad = torch.autograd.grad(loss, base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - args.base_lr * p[0], zip(grad, base_learner.parameters())))

        predicts_q = base_learner(vector, fast_weights)
        true = attack.cpu().tolist()

        for _ in range(1, args.update_step):
            predicts = base_learner(out_vectors, fast_weights)
            loss = F.cross_entropy(predicts, attack)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - args.base_lr * p[0], zip(grad, fast_weights)))
            predicts_q = base_learner(out_vectors, fast_weights)
        
        preds = predicts_q.argmax(dim=1).cpu().tolist()
        test_f1 = f1_score(true,preds,average = 'weighted')
        test_nmi = normalized_mutual_info_score(true, preds)
        nmi += test_nmi
        f1 += test_f1
        precision, recall, _, _ = precision_recall_fscore_support(true, preds, average='weighted')
        precision_score.append(precision)
        recall_score.append(recall)
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
        # f1 = 0
    for i in range(10):
        f1 = meta_train()
        print(f'average = {f1}')
        test, nmi = meta_test()
        print(f'test = {test}, nmi = {nmi}')
        test_f1_scores.append(test)
        test_nmi_scores.append(nmi)
    # print(f1)
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
    with open('results_NFCSE_mtl.txt', 'w') as file:
        print(f'Average Precision: {avg_precision:.4f} ± {std_precision:.4f}', file = file)
        print(f'Average Recall: {avg_recall:.4f} ± {std_recall:.4f}', file = file)
        print(f'Average F1 Score: {avg_f1:.4f} ± {std_f1:.4f}', file = file)
        print(f'Average NMI: {avg_nmi:.4f} ± {std_nmi:.4f}', file = file)
    






