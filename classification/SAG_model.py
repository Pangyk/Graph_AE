from abc import ABC
from torch_sparse import spspmm, coalesce
from torch_geometric.nn import SAGPooling, SAGEConv
from torch_geometric.utils import sort_edge_index, add_remaining_self_loops
import torch.nn.functional as F
import torch


class Net(torch.nn.Module, ABC):

    def __init__(self, input_size, depth, rate, shapes, device):
        super(Net, self).__init__()
        self.device = device
        self.depth = depth
        self.direction = 1
        self.down_list = torch.nn.ModuleList()
        self.up_list = torch.nn.ModuleList()
        self.pool_list = torch.nn.ModuleList()
        # encoder
        conv = SAGEConv(input_size, shapes[0])
        self.down_list.append(conv)
        for i in range(self.depth - 1):
            pool = SAGPooling(shapes[i], rate[i])
            self.pool_list.append(pool)
            conv = SAGEConv(shapes[i], shapes[i + 1])
            self.down_list.append(conv)
        pool = SAGPooling(shapes[-1], rate[-1])
        self.pool_list.append(pool)

        # decoder
        for i in range(self.depth - 1):
            conv = SAGEConv(shapes[self.depth - i - 1], shapes[self.depth - i - 2])
            self.up_list.append(conv)
        conv = SAGEConv(shapes[0], input_size)
        self.up_list.append(conv)

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        # edge_index, edge_weight = coalesce(edge_index, edge_weight, num_nodes, num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        return edge_index, edge_weight

    def forward(self, data):
        x, edge_index, y, batch = data.x, data.edge_index, data.y, data.batch
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])

        edge_list = []
        perm_list = []
        shape_list = []
        edge_weight = x.new_ones(edge_index.size(1))

        f, e, b = x, edge_index, batch
        for i in range(self.depth):
            if i < self.depth:
                edge_list.append(e)
            f = self.down_list[i](f, e)
            shape_list.append(f.shape)
            f = F.leaky_relu(f)
            f, e, _, b, perm, _ = self.pool_list[i](f, e, batch=b)
            if i < self.depth - 1:
                e, edge_weight = self.augment_adj(e, edge_weight, f.shape[0])
            perm_list.append(perm)
        latent_x, latent_edge = f, e

        z = f
        for i in range(self.depth):
            index = self.depth - i - 1
            shape = shape_list[index]
            up = torch.zeros(shape).to(self.device)
            p = perm_list[index]
            up[p] = z
            z = self.up_list[i](up, edge_list[index])
            if i < self.depth - 1:
                z = torch.relu(z)

        edge_list.clear()
        perm_list.clear()
        shape_list.clear()

        return z, latent_x, latent_edge, b
