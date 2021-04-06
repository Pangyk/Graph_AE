from abc import ABC

from torch_geometric.nn import global_mean_pool, SAGEConv
import torch.nn.functional as F
import torch


class MLP(torch.nn.Module, ABC):

    def __init__(self, input_size, hidden, num_classes, dropout):
        super(MLP, self).__init__()
        self.training = False
        self.dropout = dropout
        self.conv1 = SAGEConv(input_size, input_size // 2)
        self.conv2 = SAGEConv(input_size // 2, input_size // 2)
        self.conv3 = SAGEConv(input_size // 2, input_size // 2)

        self.lin1 = torch.nn.Linear(input_size // 2, hidden)
        self.bn1 = torch.nn.BatchNorm1d(hidden)
        self.lin2 = torch.nn.Linear(hidden, hidden // 2)
        self.bn2 = torch.nn.BatchNorm1d(hidden // 2)
        self.lin3 = torch.nn.Linear(hidden // 2, hidden // 4)
        self.bn3 = torch.nn.BatchNorm1d(hidden // 4)
        self.lin4 = torch.nn.Linear(hidden // 4, num_classes)

    def forward(self, x, edge_index, batch):
        x1 = self.conv1(x, edge_index)
        x1 = self.conv2(x1, edge_index)
        x1 = self.conv3(x1, edge_index)
        c = global_mean_pool(x1, batch)
        c = F.dropout(c, p=self.dropout, training=self.training)
        c = self.lin1(c)
        b = self.bn1(c)
        b = torch.tanh(b)
        b = F.dropout(b, p=self.dropout, training=self.training)
        c = self.lin2(b)
        b = self.bn2(c)
        b = torch.tanh(b)
        b = F.dropout(b, p=self.dropout, training=self.training)
        c = self.lin3(b)
        b = self.bn3(c)
        b = torch.tanh(b)
        c = self.lin4(b)

        return c
