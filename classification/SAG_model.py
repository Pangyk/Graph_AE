from abc import ABC

from torch_sparse import spspmm, coalesce
from torch_geometric.nn import SAGPooling, SAGEConv as conv
from torch_geometric.utils import sort_edge_index, add_remaining_self_loops
import torch

device = torch.device('cuda')


class Net(torch.nn.Module, ABC):
    model = None
    optimizer = None

    cfg = {'lr': 1e-3,
           'betas': (0.9, 0.999)}

    def __init__(self):
        super(Net, self).__init__()
        shape = [500, 64, 64, 64]
        rate = [0.8, 0.8, 0.8]
        self.conv1 = conv(shape[0], shape[1])
        self.pool1 = SAGPooling(shape[1], rate[0])
        self.conv2 = conv(shape[1], shape[2])
        self.pool2 = SAGPooling(shape[2], rate[1])
        self.conv3 = conv(shape[2], shape[3])
        self.pool3 = SAGPooling(shape[3], rate[2])

        self.depth = 3
        self.up_list = torch.nn.ModuleList()
        for i in range(self.depth):
            self.up_list.append(conv(shape[self.depth - i], shape[self.depth - i - 1]))

    @staticmethod
    def get_instance():
        if Net.model is None:
            Net.model = Net()
            Net.optimizer = torch.optim.Adam(Net.model.parameters(), lr=Net.cfg['lr'], betas=Net.cfg['betas'])
        return Net.model

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = coalesce(edge_index, edge_weight, num_nodes, num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        return edge_index, edge_weight

    def forward(self, data):
        x, edge_index, y, batch = data.x, data.edge_index, data.y, data.batch
        # x = F.normalize(x, p=1., dim=-1)
        self.x = x

        edge_index, _ = add_remaining_self_loops(edge_index)

        edge_list = [edge_index]
        perm_list = []
        x1 = self.conv1(x, edge_index)
        shape_list = [x1.shape]
        x1, e1, _, b1, p1, _ = self.pool1(x1, edge_index, batch=batch)
        e1, _ = self.augment_adj(e1, None, x.size(0))
        edge_list += [e1]
        perm_list += [p1]
        shape_list += [x1.shape]
        x2 = self.conv2(x1, e1)
        x2, e2, _, b2, p2, _ = self.pool2(x2, e1, batch=b1)
        e2, _ = self.augment_adj(e2, None, x.size(0))
        edge_list += [e2]
        perm_list += [p2]
        shape_list += [x2.shape]
        x3 = self.conv3(x2, e2)
        x3, e3, _, b3, p3, _ = self.pool3(x3, e2, batch=b2)
        perm_list += [p3]

        z = x3
        for i in range(self.depth):
            index = self.depth - i - 1
            shape = shape_list[index]
            up = torch.zeros(shape).to(device)
            p = perm_list[index]
            up[p] = z
            z = self.up_list[i](up, edge_list[index])
            if i < self.depth - 1:
                z = torch.relu(z)

        return z, x3, e3, b3

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

                mse_loss = torch.nn.MSELoss()(z, data.x)
                mix_loss = mse_loss
                mix_loss.backward()
                reconstruction_loss += mse_loss.item()

                optimizer.step()
                nodes_num = 0
                r_graph = 0

            for data in valid_set:
                data = data.to(device)
                z, _, _, _ = model(data)
                mse_loss = torch.nn.MSELoss()(z, data.x)
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

        torch.save(model.state_dict(), "../data/model/G/" + m_name)
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
