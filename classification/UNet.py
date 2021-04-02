from abc import ABC
import torch.nn.functional as F
from graph_ae.UNet import GraphUNet
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
import torch

device = torch.device('cuda:1')


class Net(torch.nn.Module, ABC):
    model = None
    optimizer = None

    cfg = {'lr': 1e-3,
           'betas': (0.9, 0.999)}

    def __init__(self):
        super(Net, self).__init__()
        self.conv = GraphUNet(500, 64, 500, 3, 0.8, True, act=F.relu).to(device)

    @staticmethod
    def get_instance():
        if Net.model is None:
            Net.model = Net()
            Net.optimizer = torch.optim.Adam(Net.model.parameters(), lr=Net.cfg['lr'], betas=Net.cfg['betas'])
        return Net.model

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self.x = x

        d, latent_x, latent_edge, batch = self.conv(x, edge_index, batch)
        d = d.to(device)
        return d, latent_x, latent_edge, batch

    def train_model(self, train_set, train_set2, valid_set, num_epoch, m_name):
        model = self.model
        optimizer = self.optimizer
        mse_list = []
        num_nodes_list = []
        total_loss_list = []
        tmp_list = []

        for e in range(num_epoch):
            reconstruction_loss = 0
            reconstruction_loss_1 = 0
            nodes_num = 0
            r_graph = 0
            for data in train_set:
                optimizer.zero_grad()
                data = data.to(device)
                z, _, _, _ = model(data)

                mse_loss = torch.nn.MSELoss()(z, self.x)
                mix_loss = mse_loss
                mix_loss.backward()

                reconstruction_loss += mse_loss.item()
                optimizer.step()
                nodes_num += 0
                r_graph += 0

            for data in valid_set:
                data = data.to(device)
                z, _, _, _ = model(data)
                mse_loss = torch.nn.MSELoss()(z, self.x)
                reconstruction_loss_1 += mse_loss.item()

            reconstruction_loss /= len(train_set)
            reconstruction_loss_1 /= len(valid_set)

            if e >= 0:
                mse_list.append(reconstruction_loss)
                num_nodes_list.append(nodes_num)
                total_loss_list.append(reconstruction_loss)
                tmp_list.append(reconstruction_loss_1)

            print()
            print('Epoch: {:03d}'.format(e))
            print('AVG Reconstruction Loss:', reconstruction_loss)
            print('AVG test Loss:', reconstruction_loss_1)
            print('Nodes num:', nodes_num)
            print('Remaining graph:', r_graph)

        torch.save(model.state_dict(), "../data/model/U/" + m_name)
        print("model saved")
        x1, e1, batch1 = None, None, None
        for data in train_set2:
            data = data.to(device)
            _, x1, e1, batch1 = model(data)
        mean = torch.mean(x1, dim=0)
        std = torch.std(x1, dim=0) + 1e-12
        x1 = (x1 - mean) / std

        x2, e2, batch2 = None, None, None
        for data in valid_set:
            data = data.to(device)
            _, x2, e2, batch2 = model(data)
        x2 = (x2 - mean) / std

        return [mse_list, num_nodes_list, total_loss_list, tmp_list], \
               [x1.detach(), e1.detach(), batch1.detach()], \
               [x2.detach(), e2.detach(), batch2.detach()]
