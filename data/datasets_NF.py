import datetime
import ipaddress
import os
import time
from abc import ABC
from bisect import bisect_left
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset, TemporalData, Data
from z3 import Optimize, RealVector, Sum, And, sat

#Deal with datasets
def count_attacks_and_filter(csv_file, output_file):
    df = pd.read_csv(csv_file)
    if 'Attack' not in df.columns:
        print(f"{csv_file} is not exist")
        return
    #define the column you want to change
    columns_to_check = ['IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT']
    mask = ~((df[columns_to_check] == '0.0.0.0').any(axis=1) | (df[columns_to_check] == 0).any(axis=1))
    df_filtered = df[mask]
    attack_counts = df_filtered['Attack'].value_counts()
    df_filtered.to_csv(output_file, index=False)

    return attack_counts

input_csv = "./data/NF-BoT-IoT-v2.csv"
output_csv = "./data/NF-BoT-IoT-v2_dealed.csv"

print(count_attacks_and_filter(input_csv, output_csv))


class My_Dataset(Dataset, ABC):
    def __init__(self, root='./data/', transform=None, pre_transform=None):
        super(My_Dataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return 'NF-BoT-IoT-v2_dealed.csv'

    @property
    def processed_file_names(self):
        return 'data/NF-BoT-IoT-v2.pt'

    def process(self):
        DISENTANGLE = False
        if os.path.exists(f'./data/{self.processed_file_names}'):
            print(f'Path ./data/{self.processed_file_names} already existed.')
            return
        else:
            print("Preparing dataset.")
            df = pd.read_csv(f'./data/{self.raw_file_names}')
            print("Read csv done.")
            #time
            num_records = len(df)
            random_seconds = np.random.randint(0, 86400, size=num_records, dtype='int')
            random_seconds = random_seconds.tolist()
            random_time_stamps = [datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=i) for i in random_seconds]
            random_time_stamps_str = [time.strftime('%d/%m/%Y %I:%M:%S %p') for time in random_time_stamps]
            df.drop(df.columns[0], axis=1)
            df.insert(loc=0, column='src', value=0)
            # df = df.drop(columns=["Flow ID"])
            src_matches = df['IPV4_SRC_ADDR'].str.endswith(('.0', '.1', '.255'))
            dst_matches = df['IPV4_DST_ADDR'].str.endswith(('.0', '.1', '.255'))
            df["src"] = df.apply(lambda x: self.addr2num(x['IPV4_SRC_ADDR'], x['L4_SRC_PORT']), axis=1)
            df.insert(loc=1, column='dst', value=0)
            df["dst"] = df.apply(lambda x: self.addr2num(x['IPV4_DST_ADDR'], int(x['L4_DST_PORT'])), axis=1)
            df = df.drop(columns=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT'])
            #temp_data = df.pop("timestamp")
            df.insert(2, 'timestamp', random_time_stamps_str)
            temp_data = df.pop("Label")
            df.insert(3, 'state_label', temp_data)
            attack = df['Attack']
            df.drop(columns=['Attack'], inplace=True)
            print(pd.Categorical(attack).categories)
            attack = pd.Categorical(attack).codes
            df.insert(4, 'attack', attack)
            opt = list(df.columns.values)[4:]
            for name in opt:
                print(name)
                M = df[name].max()
                m = df[name].min()
                df[name] = df[name].apply(lambda x: ((x - m) / (M - m)) if (M - m) != 0 else 0)
            print("regularization done.")
            df.insert(5, 'layer i', 0)
            df.loc[src_matches, 'layer_i'] = 1
            df.insert(6, 'layer j', 0)
            df.loc[dst_matches, 'layer_j'] = 1
            temp_data = df.pop('FLOW_DURATION_MILLISECONDS')
            df.insert(7, 'FLOW_DURATION_MILLISECONDS', temp_data)
            #df['timestamp'] = df['timestamp'].apply(lambda x: (x - pd.Timestamp('1970-01-01')).total_seconds())
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
            # num_rows = len(df) 
            # t = torch.arange(0, num_rows)
            t = torch.tensor(df['timestamp'].values.tolist())
            attack = torch.tensor(df['attack'].values.tolist())
            dt = torch.tensor(df['FLOW_DURATION_MILLISECONDS'].values.tolist())
            sdf = df.iloc[:, 8:]
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
            torch.save(events, './data/NF-BoT-IoT-v2_3d.pt')
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


def main(self=My_Dataset()):
    pass

if __name__ == '__main__':
    main()
# dataset = ToNDataset()
