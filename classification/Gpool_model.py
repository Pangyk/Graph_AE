from abc import ABC
import torch.nn.functional as F
from graph_ae.UNet2 import GraphUNet
from torch_geometric.utils import (add_self_loops, remove_self_loops)
import torch


class Net(torch.nn.Module, ABC):
    def __init__(self, input_size, depth, rate, shapes, device):
        super(Net, self).__init__()
        self.device = device
        self.conv = GraphUNet(input_size, shapes, input_size, depth, rate, True, act=F.relu).to(self.device)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        d, latent_x, latent_edge, batch = self.conv(x, edge_index, batch)
        d = d.to(self.device)
        return d, latent_x, latent_edge, batch
