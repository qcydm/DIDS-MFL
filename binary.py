import time
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.data import Data
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)

# from datasets import ToNDataset
from model.MGD import MGD
from utils.LOSS_binary import Loss
from utils.MLP import MLPPredictor
from utils.funcs import *
from sklearn.metrics import f1_score, confusion_matrix
from torch_geometric.data import TemporalData

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# data = ToNDataset()
# data = data.get()
# unknown load
# data = torch.load("./data/Unknown_attacks/filtered_attack_1_0_no1.pt")

# known load
data = torch.load("./data/CIC-ToN-IoT.pt")

# data = data.to(device)

min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

attack = data.attack.cpu().numpy()

# 划分训练集、验证集和测试集
train_val_idx, test_idx = train_test_split(
    np.arange(len(attack)),
    test_size=0.15,
    stratify=attack,
    random_state=42
)
train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=0.15 / (1 - 0.15),
    stratify=attack[train_val_idx],
    random_state=42
)

# 创建新的 TemporalData 对象来存储划分后的数据
def create_new_data(data, idx):
    return TemporalData(
        src=data.src[idx],
        dst=data.dst[idx],
        t=data.t[idx],
        msg=data.msg[idx],
        label=data.label[idx],
        attack=data.attack[idx]
    )

train_data = create_new_data(data, train_idx)
val_data = create_new_data(data, val_idx)
test_data = create_new_data(data, test_idx)

# 打印各类别的数量分布
print("Train attack counts:", torch.bincount(train_data.attack.cpu()))
print("Validation attack counts:", torch.bincount(val_data.attack.cpu()))
print("Test attack counts:", torch.bincount(test_data.attack.cpu()))

# train_data, test_data, val_data = data.train_val_test_split(val_ratio=0.15, test_ratio=0.15)



train_loader = TemporalDataLoader(train_data, batch_size=200)
test_loader = TemporalDataLoader(test_data, batch_size=200)
val_loader = TemporalDataLoader(val_data, batch_size=200)
# unknown specific, unknown loader
# un_data=torch.load("./data/Unknown_attacks/filtered_attack_1_0.pt")
# print("unknown attack counts:", torch.bincount(un_data.attack.cpu()))
# unknown_loader= TemporalDataLoader(un_data,batch_size=200)
# test specific, test loader
test_loader = TemporalDataLoader(data, batch_size=200)
neighbor_loader = LastNeighborLoader(data.num_nodes, size=10, device=device)
memory_dim = time_dim = embedding_dim = 128
layer = 1
gind_params = {'num_layers': 1, 'alpha': 0.02, 'hidden_channels': 128, 'drop_input': True, 'dropout_imp': 0.5,
               'dropout_exp': 0.0, 'iter_nums': [36, 4], 'linear': True, 'double_linear': True, 'act_imp': 'tanh',
               'act_exp': 'elu', 'rescale': True, 'residual': True, 'norm': 'LayerNorm', 'final_reduce': None}

memory = TGNMemory(
    data.num_nodes,
    data.msg.size(-1),
    memory_dim,
    time_dim,
    message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
    aggregator_module=LastAggregator(),
).to(device)

mgd = MGD(in_channels=embedding_dim, out_channels=embedding_dim, **gind_params).to(device)
# load 3D-IDS model
# mgd.load_state_dict(torch.load("./data/ToN_not_2_new.pth"))

bin_predictor = MLPPredictor(in_features=embedding_dim, out_classes=2).to(device)
mul_predictor = MLPPredictor(in_features=embedding_dim, out_classes=10).to(device)

optimizer = torch.optim.Adam(
    set(memory.parameters()) | set(mgd.parameters())
    | set(bin_predictor.parameters()) | set(mul_predictor.parameters()), lr=0.0001)
criterion = Loss(2, 10)
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)


def train():
    memory.train()
    mgd.train()
    bin_predictor.train()
    mul_predictor.train()
    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.
    total_loss = 0
    for batch in train_loader:
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
        bin_out = bin_predictor(z[assoc[src]], z[assoc[dst]])
        # mul_out = mul_predictor(z[assoc[src]], z[assoc[dst]])
        loss = criterion(bin_out,label,attack,z)
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)
        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss) * batch.num_events
    return total_loss / train_data.num_events

@torch.no_grad()
def test(loader):
    memory.eval()
    mgd.eval()
    bin_predictor.eval()
    torch.manual_seed(12345)
    aps, aucs, f1s = [], [], []
    preds = []
    trues = []
    embeddings = []
    total_per_attack = defaultdict(int)
    detected_per_attack = defaultdict(int)
    all_attack = []
    all_pred = []
    all_label = []
    all_attack_pred = []
    for batch in loader:
        batch = batch.to(device)
        src, dst, t, msg, label = batch.src, batch.dst, batch.t, batch.msg, batch.label
        n_id = torch.cat([src, dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        z, last_update = memory(n_id)
        ed, m = nodeMap(torch.stack((src, dst), dim=0))
        ed = ed.to(device)
        norm_factor, ed = cal_norm(ed, num_nodes=len(z), self_loop=False)
        z = mgd(z, ed, norm_factor).to(device)
        out = bin_predictor(z[assoc[src]], z[assoc[dst]]).argmax(1)
        y_pred = out.cpu()
        y_true = label.cpu()
        trues += y_true
        preds += y_pred

        # ----------------特定攻击检测率统计-----------------
        attack_np = batch.attack.cpu().numpy()
        for a in range(1, 10):
            mask = attack_np == a
            total_per_attack[a] += mask.sum()
            detected_per_attack[a] += ((out.cpu().numpy() == 1) & mask).sum()
        # ----------------------------------------------
        # ---------------特定攻击检测F1统计------------------
        all_attack.extend(batch.attack.cpu().numpy())
        all_pred.extend(out.cpu().numpy())
        # --------------------------------------------------
        # representation for heatmap
        embeddings.append(z.cpu().numpy())  # 获取嵌入
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)
    embeddings = np.concatenate(embeddings, axis=0) #合并嵌入
    f1ss = f1_score(trues,preds)
    apss = average_precision_score(trues,preds)
    aucss = roc_auc_score(trues,preds)

    # 计算 FPR
    tn, fp, fn, tp = confusion_matrix(trues, preds).ravel()
    fpr = fp / (fp + tn)  # FPR = FP / (FP + TN)
    # 计算攻击检测率
    detection_rates = {
    a: detected_per_attack[a] / total_per_attack[a] if total_per_attack[a] else 0.0
    for a in range(1, 10)
}
    f1_per_attack = {}
    for a in range(1, 10):
        mask = np.array(all_attack) == a
        if mask.sum() == 0:
            f1_per_attack[a] = 0.0
        else:
            y_pred_a = np.array(all_pred)[mask]
            f1_per_attack[a] = f1_score(np.ones_like(y_pred_a), y_pred_a)
    return apss,aucss,f1ss,embeddings,fpr,f1_per_attack

def plot_correlation_heatmap(correlation_matrix):

    cmap = sns.color_palette("Blues", as_cmap=True)
    
    plt.figure(figsize=(10, 8))
    # 绘制热力图
    heatmap = sns.heatmap(correlation_matrix, annot=False, cmap=cmap, center=0, square=True, cbar=True)
    plt.title('Correlation Heatmap of Embeddings')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Embedding Dimension')
    
    # 调整颜色条的标签
    cbar = heatmap.collections[0].colorbar
    cbar.set_ticks([correlation_matrix.min(), 0, correlation_matrix.max()])
    cbar.set_ticklabels([f'{correlation_matrix.min():.2f}', '0', f'{correlation_matrix.max():.2f}'])
    plt.savefig('./visualization_DIDS_TON', dpi=300, bbox_inches='tight')
    plt.show()

######主函数

def main():
    for epoch in range(1, 11):
        loss = train()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
        st = time.time()
        # known test
        # test_loader
        test_ap, test_auc, test_f1 , embeddings, test_fpr, f1_per_attack = test(test_loader)
        # unknown test
        # unknown_loader
        # test_ap, test_auc, test_f1 , embeddings, test_fpr, f1_per_attack = test(unknown_loader)
        ft = time.time()
        dt = ft - st
        print(f'Test time for epoch {epoch:02d}: {dt}, avrage: {dt / 200}')
        print(f'Test AP: {test_ap:.4f}, Test F1: {test_f1:.4f}, Test AUC: {test_auc:.4f}, Test FPR: {test_fpr:.4f}')
        #-------------------- Per attack detection rate and F1 score -----------------------------------------
        # for a in range(1, 10):
        #     print(f"Attack-{a} Detection Rate: {dr[a]:.4f}")
        # for a in range(1, 10):
        #     print(f"Attack-{a} F1: {f1_per_attack[a]:.4f}")
        #-----------------------------------------------------------------------------------------------------
        # Depict correlation heatmap
        # if epoch == 1:
        #     correlation_matrix = np.corrcoef(embeddings, rowvar=False)
        #     plot_correlation_heatmap(correlation_matrix)




if __name__ == '__main__':
    main()
