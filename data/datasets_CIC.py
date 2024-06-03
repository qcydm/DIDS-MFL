import ipaddress
import os
import time
from abc import ABC
from bisect import bisect_left
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset, TemporalData, Data

#Deal with datasets
def count_attacks_and_filter(csv_file, output_file):
    df = pd.read_csv(csv_file)
    if 'Attack' not in df.columns:
        print(f"{csv_file} is not exist")
        return
    #define the column you want to change
    columns_to_check = ['Src IP', 'Src Port', 'Dst IP', 'Dst Port']
    mask = ~((df[columns_to_check] == '0.0.0.0').any(axis=1) | (df[columns_to_check] == 0).any(axis=1))
    df_filtered = df[mask]
    attack_counts = df_filtered['Attack'].value_counts()
    df_filtered.to_csv(output_file, index=False)

    return attack_counts

input_csv = "./data/CIC-ToN-IoT.csv"
output_csv = "./data/CIC-ToN-IoT_dealed.csv"
print(count_attacks_and_filter(input_csv, output_csv))

class My_Dataset(Dataset, ABC):
    def __init__(self, root='./data/', transform=None, pre_transform=None):
        super(My_Dataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return 'CIC-ToN-IoT_dealed.csv'

    @property
    def processed_file_names(self):
        return 'CiC-ToN-IoT.pt'

    def process(self):
        if os.path.exists(f'./data/{self.processed_file_names}'):
            print(f'Path ./data/{self.processed_file_names} already existed.')
            return
        else:
            print("Preparing dataset CiC-ToN-IoT.")
            df = pd.read_csv(f'./data/{self.raw_file_names}')
            print("Read csv done.")
            df = df.drop(columns=["Flow ID"])
            df.insert(loc=0, column='src', value=0)
            df["src"] = df.apply(lambda x: self.addr2num(x['Src IP'], int(x['Src Port'])), axis=1)
            df.insert(loc=1, column='dst', value=0)
            df["dst"] = df.apply(lambda x: self.addr2num(x['Dst IP'], int(x['Dst Port'])), axis=1)
            df = df.drop(columns=['Src IP', 'Dst IP', 'Src Port', 'Dst Port'])
            temp_data = df.pop("Timestamp")
            df.insert(2, 'timestamp', temp_data)
            temp_data = df.pop("Label")
            df.insert(3, 'state_label', temp_data)
            attack = df['Attack']
            df.drop(columns=['Attack'], inplace=True)
            print(pd.Categorical(attack).categories)
            attack = pd.Categorical(attack).codes
            df.insert(4, 'attack', attack)
            opt = list(df.columns.values)[5:]
            for name in opt:
                print(name)
                M = df[name].max()
                m = df[name].min()
                df[name] = df[name].apply(lambda x: ((x - m) / (M - m)) if (M - m) != 0 else 0)
            print("regularization done.")
            temp_data = df.pop('Flow Duration')
            df.insert(7, 'Flow Duration', temp_data)
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
            df.to_csv(f'./data/temp-{self.raw_file_names}')

            attack_types = df['attack'].unique()
            sampled_df = pd.DataFrame()
            for attack_type in attack_types:
                filtered_rows = df[df['attack'] == attack_type]
                sample_size = int(
                    len(filtered_rows) if 0.05 * len(filtered_rows) <= 1000 else 0.05 * len(filtered_rows))
                random_sample = filtered_rows.sample(n=sample_size)
                sampled_df = pd.concat([sampled_df, random_sample])

            sampled_df = sampled_df.sort_values('timestamp')
            sampled_df.to_csv(f'./data/selected-{self.raw_file_names}')
            print(sampled_df['attack'].value_counts())
            df = sampled_df
            print(pd.Categorical(df['attack']).categories)
            df['attack'] = pd.Categorical(df['attack']).codes
            print(dict(df['state_label'].value_counts()))
            df.fillna(0, inplace=True)
            src = torch.tensor(df['src'].values.tolist())
            dst = torch.tensor(df['dst'].values.tolist())
            label = torch.tensor(df['state_label'].values.tolist())
            t = torch.tensor(df['timestamp'].values.tolist())
            attack = torch.tensor(df['attack'].values.tolist())
            dt = torch.tensor(df['Flow Duration'].values.tolist())


            edge_index = torch.stack([src, dst], dim=1).T
            print(edge_index.shape)
            sdf = df.iloc[:, 8:]

            length: int = len(df)
            node_list = []
            origin_nodelist = []
            num_columns = sdf.shape[1]
            for i in range(length):
                list_i = []
                originlist_i = [src[i].item(), dst[i].item()]
                print(f'Node: {i}', originlist_i)
                for j in range(num_columns):
                    list_i.append(sdf.iloc[i, j])
                node_list.append(list_i)
                origin_nodelist.append(originlist_i)
            edge_index = []
            for i in range(length):
                src = node_list[i][0]
                dst = node_list[i][1]
                for j in range(i + 1, length):
                    src1 = node_list[j][0]
                    dst1 = node_list[j][1]
                    if (src == src1) or (src == dst1) or (dst == src1) or (dst == dst1):
                        edge_index.append([i, j])
                        edge_index.append([j, i])
            print(edge_index)
            # print(node_list)
            edge_index = torch.tensor(edge_index).t()

            msg1 = torch.tensor(node_list)
            msg = torch.tensor(sdf.values.tolist())


            events = TemporalData(
                src=src,
                dst=dst,
                t=t,
                dt=dt,
                msg=msg,
                label=label,
                attack=attack)
            data2 = Data(
                x=msg1,
                y=attack,
                edge_index=edge_index
            )
            # torch.save(events, f"./data/{self.processed_file_names}")
            torch.save(data2, './data/explain.pt')
            return

    def addr2num(self, ip, port):
        bin_ip = bin(int(ipaddress.IPv4Address(ip))).replace("0b", "").zfill(32)
        bin_port = bin(port).replace('0b', '').zfill(16)
        id = bin_ip + bin_port
        id = int(id, 2)
        return id

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
