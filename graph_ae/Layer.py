from abc import ABC

from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.nn import global_add_pool as g_pooling
from torch_geometric.utils import sort_edge_index
# from graph_ae.GATConv import GATConv
from graph_ae.SAGEAttn import SAGEAttn
from torch_sparse import spspmm
import torch.nn.functional as f
from torch.nn import Parameter
import torch

from torch_geometric.nn import SAGEConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SGAT(torch.nn.Module, ABC):

    def __init__(self, size, in_channel, out_channel, heads: int = 1):
        super(SGAT, self).__init__()
        self.size = size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.heads = heads
        # self.pm = Parameter(torch.ones([self.size]))
        self.gat_list = torch.nn.ModuleList()

        for i in range(size):
            self.gat_list.append(SAGEAttn(in_channel, out_channel).to(device))

        self.reset_parameters()

    def reset_parameters(self):
        # self.pm.data.fill_(1)
        for conv in self.gat_list:
            conv.reset_parameters()

    def forward(self, x, edge_index, direction=1):
        feature_list = None
        attention_list = []
        # pm = torch.softmax(self.pm, dim=-1)
        idx = 0
        for conv in self.gat_list:
            feature, attn = conv(x, edge_index)
            if feature_list is None:
                feature_list = f.leaky_relu(feature)
            else:
                feature_list += f.leaky_relu(feature)
            attention_list.append(attn)
            idx += 1

        attention_list = torch.stack(attention_list, dim=1)
        if attention_list.shape[1] > 1:
            attention_list = torch.sum(attention_list, dim=1)
        e_batch = edge_index[0]
        node_scores = direction * g_pooling(attention_list, e_batch).view(-1)
        return feature_list, node_scores


class SGConv(torch.nn.Module, ABC):

    def __init__(self, size, in_channel, out_channel, heads: int = 1):
        super(SGConv, self).__init__()
        self.size = size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.heads = heads
        self.gat_list = torch.nn.ModuleList()

        for i in range(size):
            self.gat_list.append(SAGEConv(in_channel, out_channel).to(device))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.gat_list:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        feature_list = None
        for conv in self.gat_list:
            feature = conv(x, edge_index)
            if feature_list is None:
                feature_list = [f.leaky_relu(feature)]
            else:
                feature_list += [f.leaky_relu(feature)]
        feature_list = torch.cat(feature_list, dim=-1)

        return feature_list


class GAEConv(torch.nn.Module, ABC):

    def __init__(self, size, in_channel, out_channel, heads: int = 1):
        super(GAEConv, self).__init__()
        self.size = size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.heads = heads
        self.gat_list = torch.nn.ModuleList()

        for i in range(size):
            self.gat_list.append(GATConv(in_channel, out_channel).to(device))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.gat_list:
            conv.reset_parameters()

    def forward(self, x, edge_index, direction=1):
        feature_list = None
        attention_list = []
        idx = 0
        for conv in self.gat_list:
            feature, (edge_id, attention_weight) = conv(x, edge_index, return_attention_weights=True)
            if feature_list is None:
                feature_list = f.leaky_relu(feature)
            else:
                feature_list += f.leaky_relu(feature)
            attention_list.append(attention_weight.view(-1))
            idx += 1

        attention_cat = torch.stack(attention_list, dim=1)
        if attention_cat.shape[1] > 1:
            attention_cat = attention_cat.sum(dim=1)

        e_batch = edge_index[0]
        node_scores = direction * g_pooling(attention_cat, e_batch).view(-1)

        return feature_list, node_scores


class Pooling(torch.nn.Module, ABC):

    def __init__(self, rate):
        super(Pooling, self).__init__()
        self.rate = rate

    def forward(self, x, edge_index, attention, batch=None, direction=1):
        e_batch = edge_index[0]
        degree = torch.bincount(e_batch)
        node_scores = direction * g_pooling(attention, e_batch).view(-1)
        node_scores = node_scores.mul(degree)

        perm = topk(node_scores, self.rate, batch)

        edge_index, _ = self.augment_adj(edge_index, None, x.size(0))
        edge_index, _ = filter_adj(edge_index, None, perm, num_nodes=node_scores.size(0))
        x = x[perm]
        batch = batch[perm]

        return x, edge_index, batch, perm.view((1, -1))

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        return edge_index, edge_weight
