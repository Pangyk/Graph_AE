from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import sort_edge_index
from torch_geometric.datasets import TUDataset
from torch.utils.data import Dataset
from tqdm import tqdm
import os.path as osp
from abc import ABC
import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch_sparse import coalesce
import json

from sklearn.preprocessing import LabelEncoder
import pandas as pd


class CitePaper(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(CitePaper, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['pubmed.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []
        node_dict = dict()
        feature_dict = dict()

        with open("data/Pubmed-Diabetes/node.tab", "r") as f:
            s_list = f.readlines()
            sl = s_list[1].split('\t')
            count = 0
            for s in sl:
                if s.startswith("numeric:"):
                    s_len = len(s)
                    feature_dict[s[8: s_len - 4]] = count
                    count += 1

            nodes = np.zeros((len(s_list) - 2, 500))
            for i in range(2, len(s_list)):
                sl = s_list[i].split('\t')
                for s in sl:
                    if s.startswith("w-"):
                        lbs = s.split("=")
                        node_dict[sl[0]] = i - 2
                        nodes[i - 2][feature_dict[lbs[0]]] = float(lbs[1])

            x = torch.from_numpy(nodes).float()

        with open("data/Pubmed-Diabetes/edge.tab", "r") as f:
            s_list = f.readlines()
            edges = np.zeros((2, len(s_list)), dtype=np.long)
            index = 0
            for sl in s_list:
                s = sl.strip('\n').split('\t')
                edges[0][index] = node_dict[s[1].split(":")[1]]
                edges[1][index] = node_dict[s[3].split(":")[1]]
                index += 1

            edge_index = torch.from_numpy(edges).long()

        data_list.append(Data(x=x, edge_index=edge_index, num_nodes=x.shape[0]))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SellGraphs(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SellGraphs, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['buy.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []

        df = pd.read_csv('data/buy/yoochoose-clicks.dat', header=None)
        df.columns = ['session_id', 'timestamp', 'item_id', 'category']

        buy_df = pd.read_csv('data/buy/yoochoose-buys.dat', header=None)
        buy_df.columns = ['session_id', 'timestamp', 'item_id', 'price', 'quantity']

        item_encoder = LabelEncoder()
        df['item_id'] = item_encoder.fit_transform(df.item_id)

        sampled_session_id = np.random.choice(df.session_id.unique(), 100000, replace=False)
        df = df.loc[df.session_id.isin(sampled_session_id)]

        df['label'] = df.session_id.isin(buy_df.session_id)
        # process by session_id
        grouped = df.groupby('session_id')
        for session_id, group in tqdm(grouped):
            sess_item_id = LabelEncoder().fit_transform(group.item_id)
            group = group.reset_index(drop=True)
            group['sess_item_id'] = sess_item_id
            node_features = group.loc[group.session_id == session_id, ['sess_item_id', 'item_id']].sort_values(
                'sess_item_id').item_id.drop_duplicates().values

            node_features = torch.LongTensor(node_features).unsqueeze(1)
            target_nodes = group.sess_item_id.values[1:]
            source_nodes = group.sess_item_id.values[:-1]

            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            x = node_features

            y = torch.FloatTensor([group.label.values[0]])

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class BasicGraphs(InMemoryDataset, ABC):
    def __init__(self, root, transform=None, pre_transform=None):
        self.graph_num = 100
        self.min_nodes = 10
        self.max_nodes = 20
        self.feature_size = 500
        self.max_edge_rate = 0.2
        super(BasicGraphs, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        is_processed = False

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
            is_processed = True

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            is_processed = True

        if not is_processed:
            for _ in tqdm(range(self.graph_num)):
                data_list.append(self.get_graph())

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_graph(self):
        num_nodes = np.random.randint(self.min_nodes, self.max_nodes)
        x = torch.clamp(torch.randn((num_nodes, self.feature_size)) - 0.1, min=0.0)
        max_edges = np.int(num_nodes * self.max_edge_rate) + 1
        node_list = np.arange(num_nodes)
        ns_list = []
        ne_list = []
        for i in range(num_nodes):
            num_edges = np.random.randint(1, max_edges)
            tmp_list = np.concatenate([node_list[:i], node_list[i + 1:]])
            indices = np.random.choice(tmp_list, size=num_edges, replace=False)
            for j in range(num_edges):
                ns_list.append(i)
                ne_list.append(indices[j])
        edge_index_np = np.array([ns_list, ne_list])
        edge_index = torch.from_numpy(edge_index_np).long()

        return Data(x=x, edge_index=edge_index)


class ExistGraph(Dataset):

    @staticmethod
    def get_data_set(data_name):
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'MUTAG')
        data_set = TUDataset(path, name=data_name, use_node_attr=True)
        return data_set


class SelectGraph(InMemoryDataset, ABC):
    data_name = ''
    thresh = 0
    direction = 0

    def __init__(self, root, transform=None, pre_transform=None):
        super(SelectGraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        print(SelectGraph.direction)
        data_list = []
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'MUTAG')
        data_set = TUDataset(path, name=SelectGraph.data_name, use_node_attr=True)
        for data in data_set:
            if SelectGraph.direction < 0:
                if data.y < SelectGraph.thresh:
                    data_list.append(data)
            else:
                if data.y >= SelectGraph.thresh:
                    data_list.append(data)

        random.shuffle(data_list)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SceneGraphs(InMemoryDataset, ABC):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SceneGraphs, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['scene.pt']

    def process(self):
        data_list = []

        with open('data/scene/objects.json', 'r') as f:
            objects = f.read()
        with open('data/scene/relationships.json', 'r') as f:
            links = f.read()

        object_entry = json.loads(objects)
        link_entry = json.loads(links)
        word_dict = dict()

        count = 0
        num_samples = 5000
        feature_size = 20

        for i in tqdm(range(num_samples)):
            objs = object_entry[i]["objects"]
            for obj in objs:
                name = obj["names"][0]
                if name not in word_dict.keys():
                    word_dict[name] = count
                    count += 1

        embeds = nn.Embedding(len(word_dict), feature_size)
        eb = embeds(Variable(torch.arange(0, len(word_dict)).long()))
        for i in tqdm(range(num_samples)):
            objs = object_entry[i]["objects"]
            id_dict = dict()
            node_list = []
            idx = 0
            if len(objs) == 0:
                continue
            for obj in objs:
                name = obj["names"][0]
                node_list.append(eb[word_dict[name]])
                id_dict[obj["object_id"]] = idx
                for j in obj["merged_object_ids"]:
                    id_dict[j] = idx
                idx += 1
            x = torch.stack(node_list)
            print()
            print(x.shape)

            from_list = []
            to_list = []
            links = link_entry[i]["relationships"]
            for link in links:
                v = link["object"]["object_id"]
                u = link["subject"]["object_id"]
                if v in id_dict.keys() and u in id_dict.keys():
                    from_list.append(id_dict[v])
                    to_list.append(id_dict[u])
            edge_index = torch.tensor([from_list, to_list], dtype=torch.long)
            edge_index, _ = sort_edge_index(edge_index, None, x.shape[0])
            if len(from_list) > 0:
                edge_index, _ = coalesce(edge_index, None, x.shape[0], x.shape[0])
                print(edge_index.shape)
                print()
                data = Data(x=x, edge_index=edge_index)
                data_list.append(data)

        print(len(word_dict))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
