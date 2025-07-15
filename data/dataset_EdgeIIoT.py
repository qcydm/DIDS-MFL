import datetime
import ipaddress
import os
import re
import time
from abc import ABC
from bisect import bisect_left

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset, TemporalData, Data
from z3 import Optimize, RealVector, Sum, And, sat


df = pd.read_csv("./data/DNN-EdgeIIoT-dataset.csv", low_memory=False)
# df = df[:10000]
def is_valid_ip(ip_addr):
    if pd.isnull(ip_addr):
        return False
    return re.match(r'^(\d{1,3}\.){3}\d{1,3}$', ip_addr) is not None

df = df[(df['ip.src_host'].apply(is_valid_ip)) & (df['ip.dst_host'].apply(is_valid_ip)) & pd.notnull(df['frame.time'])]

def remove_last_three_digits(time_str):
    if isinstance(time_str, str) and len(time_str) > 3:
        return time_str[:-3]  
    else:
        return time_str 

df['frame.time'] = df['frame.time'].apply(remove_last_three_digits)

# 保存到新的 CSV 文件
df.to_csv("./data/DNN-EdgeIIoT-dataset_dealed.csv", index=False)

# 更改
def safe_strptime(x):
    try:
        return int(datetime.datetime.strptime(x, "%d/%m/%Y %I:%M:%S %p").timestamp())
    except ValueError:
        print("ValueError")
        return None  # 或者你可以选择返回一个默认值，或者记录错误

class My_Dataset(Dataset, ABC):
    def __init__(self, root='./data/', transform=None, pre_transform=None):
        super(My_Dataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return 'DNN-EdgeIIoT-dataset_dealed.csv'

    @property
    def processed_file_names(self):
        return 'DNN-EdgeIIoT-dataset.pt'


    def process(self):
        DISENTANGLE = True

        if os.path.exists(f'./data/{self.processed_file_names}'):
            print(f'Path ./data/{self.processed_file_names} already existed.')
            return
        else:
            print("Preparing dataset EdgeIIoT-dataset.")
            df = pd.read_csv(f'./data/{self.raw_file_names}', low_memory=False)
            print("Read csv done.")


            #time
            
            num_records = len(df)
            print(num_records)
# OverflowError: mktime argument out of range 
            # 于是更改
            # 限制时间戳在合理的范围内，mktime，在中国大陆（+0800时区），只能接收大于对应1970年1月1日，8时0分0秒的时间戳
            max_seconds = (datetime.datetime(2030, 1, 1) - datetime.datetime(1970, 1, 2)).total_seconds()
            random_seconds = np.random.randint(0, int(max_seconds), size=num_records, dtype='int')
            # random_seconds = np.random.randint(0, 86400, size=num_records, dtype='int')
            random_seconds = random_seconds.tolist()
            random_time_stamps = [datetime.datetime(1970, 1, 2) + datetime.timedelta(seconds=i) for i in random_seconds]
            random_time_stamps_str = [time.strftime('%d/%m/%Y %I:%M:%S %p') for time in random_time_stamps]
            print(random_time_stamps_str[:20])
            
            
            
            df.drop(df.columns[0], axis=1)
            # df = df.drop(columns=["Flow ID"])
            src_matches = df['ip.src_host'].str.endswith(('.0', '.1', '.255'))
            dst_matches = df['ip.dst_host'].str.endswith(('.0', '.1', '.255'))
            # df['tcp.srcport'] = pd.to_numeric(df['tcp.srcport'], errors='coerce').fillna(0).astype(int)
            # df['tcp.dstport'] = pd.to_numeric(df['tcp.dstport'], errors='coerce').fillna(0).astype(int)
            # print(df['ip.src_host'])
            df.insert(loc=0, column='src', value=0)
            df["src"] = df.apply(lambda x: self.addr2num(x['ip.src_host'], int(x['tcp.srcport'])), axis=1)
            df.insert(loc=1, column='dst', value=0)
            df["dst"] = df.apply(lambda x: self.addr2num(x['ip.dst_host'], int(x['tcp.dstport'])), axis=1)
            df = df.drop(columns=['ip.src_host', 'ip.dst_host', 'tcp.srcport', 'tcp.dstport'])
            # temp_data = df.pop("Timestamp")
            temp_data = df.pop("frame.time")
            df.insert(2, 'timestamp', random_time_stamps_str)
            temp_data = df.pop("Attack_label")
            df.insert(3, 'state_label', temp_data)
            attack = df['Attack_type']
            df.drop(columns=['Attack_type'], inplace=True)
            print(pd.Categorical(attack).categories)
            attack = pd.Categorical(attack).codes
            df.insert(4, 'attack', attack)
            df = df.drop(columns=['http.request.uri.query', 'http.request.method', 'http.referer', 'http.request.full_uri', 'http.request.version', 'http.response', 'dns.qry.name', 'mqtt.msg', 'mqtt.protoname', 'mqtt.topic', 'arp.dst.proto_ipv4', 'http.file_data', 'tcp.options', 'tcp.payload', 'dns.qry.name.len', 'mqtt.conack.flags'])
            opt = list(df.columns.values)[5:]
            for name in opt:
                M = df[name].max()
                m = df[name].min()
                df[name] = df[name].apply(lambda x: ((x - m) / (M - m)) if (M - m) != 0 else 0)
            print("regularization done.")
            df.insert(5, 'layer i', 0)
            df.loc[src_matches, 'layer_i'] = 1
            df.insert(6, 'layer j', 0)
            df.loc[dst_matches, 'layer_j'] = 1
            #temp_data = df.pop('Flow Duration')
            df.insert(7, 'FLOW_DURATION_MILLISECONDS', temp_data)


            # 应用 safe_strptime 函数到 'timestamp' 列
            # df['timestamp'] = df['timestamp'].apply(safe_strptime)
                # OverflowError: mktime argument out of range 
            # 于是更改为datatime
            # df['timestamp'] = df['timestamp'].apply(lambda x: int(datetime.datetime.strptime(x, "%d/%m/%Y %I:%M:%S %p").timestamp()))
                                                    

            df['timestamp'] = df['timestamp'].apply(
                lambda x: int(time.mktime(time.strptime(x, "%d/%m/%Y %I:%M:%S %p"))))
            df['timestamp'] = df['timestamp'] - df['timestamp'].min()
            print("Convert time done.")
            src_set = df.src.values
            dst_set = df.dst.values
            node_set = set(src_set).union(set(dst_set))
            ordered_node_set = sorted(node_set)
            assert (len(ordered_node_set) == len(set(ordered_node_set)))  # 查重
            df["src"] = df["src"].apply(lambda x: bisect_left(ordered_node_set, x) + 1)
            df["dst"] = df["dst"].apply(lambda x: bisect_left(ordered_node_set, x) + 1)
            print("Almost done.")
            df.sort_values(by="timestamp", inplace=True, ascending=True)
            print("Sort done.")
            df.fillna(0, inplace=True)
            df['layer_i'].value_counts()
            # df.to_csv(f'./data/temp-{self.raw_file_names}')

            attack_types = df['attack'].unique()
            sampled_df = pd.DataFrame()
            for attack_type in attack_types:
                filtered_rows = df[df['attack'] == attack_type]
                sample_size = int(
                    len(filtered_rows) if 0.05 * len(filtered_rows) <= 1000 else 0.05 * len(filtered_rows))
                random_sample = filtered_rows.sample(n=sample_size)
                sampled_df = pd.concat([sampled_df, random_sample])

            sampled_df = sampled_df.sort_values('timestamp')
            # sampled_df.to_csv(f'./data/selected-{self.raw_file_names}')
            print(sampled_df['attack'].value_counts())
            df = sampled_df
            print(pd.Categorical(df['attack']).categories)
            df['attack'] = pd.Categorical(df['attack']).codes
            print(dict(df['state_label'].value_counts()))
            df.fillna(0, inplace=True)
            src = torch.tensor(df['src'].values.tolist())
            dst = torch.tensor(df['dst'].values.tolist())
            src_layer = torch.tensor(df['layer i'].values.tolist())
            dst_layer = torch.tensor(df['layer j'].values.tolist())
            label = torch.tensor(df['state_label'].values.tolist())
            t = torch.tensor(df['timestamp'].values.tolist())
            attack = torch.tensor(df['attack'].values.tolist())
            dt = torch.tensor(df['FLOW_DURATION_MILLISECONDS'].values.tolist())
            sdf = df.iloc[:, 8:]
            for column in sdf.columns:
                sdf[column] = pd.to_numeric(sdf[column], errors='coerce')
            if sdf.isnull().any().any():
                sdf.fillna(0, inplace=True)  
            if DISENTANGLE:
                sdf_mean = sdf.mean()
                select_index = sdf_mean.sort_values().index
                select_index = select_index[:int(len(select_index) / 10)]
                sdf = sdf.apply(self.disentangle, args=(select_index,), axis=1)
            msg = torch.tensor(sdf.values.tolist())

            events = TemporalData(
                src=src,
                dst=dst,
                src_layer=src_layer,
                dst_layer=dst_layer,
                t=t,
                dt=dt,
                msg=msg,
                label=label,
                attack=attack)
            # torch.save(events, f"./data/{self.processed_file_names}")
            events.t = events.t.type(torch.int64) 
            events.dt = events.dt.type(torch.float32)
            torch.save(events, './data/DNN-EdgeIIoT-dataset.pt')
            return

    def addr2num(self, ip, port):
        bin_ip = bin(int(ipaddress.IPv4Address(ip))).replace("0b", "").zfill(32)
        bin_port = bin(port).replace('0b', '').zfill(16)
        id = bin_ip + bin_port
        id = int(id, 2)
        return id

    def solver(self, N):
        Wmin = 0
        Wmax = 1
        B = sum(N)
        s = Optimize()
        M = len(N)
        # print(M)
        W = RealVector('w', M)
        s.add(Sum([n * w for n in N for w in W]) < B)
        T = ''
        for i in range(0, M - 1):
            s.add(W[i] * N[i] <= W[i + 1] * N[i + 1])
            s.add(And(Wmin <= W[i], W[i] <= Wmax))
        for i in range(1, M - 1):
            s.add(2 * W[i] * N[i] <= W[i - 1] * N[i - 1] + W[i + 1] * N[i + 1])
            T = T + 2 * W[i] * N[i] - W[i - 1] * N[i - 1] - W[i + 1] * N[i + 1]
        s.maximize(W[M - 1] * N[M - 1] - W[0] * N[0] + T)

        if s.check() == sat:
            m = s.model()
            result = np.array(
                [float(m[y].as_decimal(10)[:-2]) if (len(m[y].as_decimal(10)) > 1) else float(m[y].as_decimal(10)) for y
                 in
                 W])

            return result

    def disentangle(self, N, select_axis):
        o = N[select_axis]
        t = self.solver(np.array(N[select_axis] + 0.01))
        if type(t) is np.ndarray and t.any() != None:
            N = N.replace(N[select_axis] + N[select_axis] * t)
            return N

    def get(self, idx=0):
        return torch.load(f'./data/{self.processed_file_names}')

    def len(self):
        pass

    def __len__(self) -> int:
        return super().__len__()


def main():
    dataset = My_Dataset()
    dataset.process()

if __name__ == '__main__':
    main()