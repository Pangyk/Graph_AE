from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATConv, GCNConv
import torch.nn.functional as F
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(torch.nn.Module):
    model = None
    optimizer = None

    cfg = {'lr': 1e-3,
           'betas': (0.9, 0.999),
           'lbd': 0.5,
           'thresh': torch.tensor(0.2).to(device),
           'p1': torch.tensor([10.0]).to(device),
           'p0': torch.tensor([100.0]).to(device),
           'h': 2000}

    def __init__(self):
        super(Net, self).__init__()
        hidden = 64
        out = 5
        # encoder
        self.conv1 = GATConv(out, hidden)
        self.conv2 = GATConv(hidden, hidden)
        self.conv3 = GATConv(hidden, hidden)
        # mask
        self.mask1 = GATConv(hidden, 32)
        self.mask2 = GATConv(32, 32)
        self.mask3 = GATConv(32, 2)
        # decoder
        self.conv4 = GATConv(hidden, hidden)
        self.conv5 = GATConv(hidden, hidden)
        self.conv6 = GATConv(hidden, out)

        self.lin1 = torch.nn.Linear(hidden, hidden)
        self.lin2 = torch.nn.Linear(hidden, 11)

        self.x_num_nodes = 0
        self.penalty = 0
        self.x = 0
        self.r_graph = 0

    @staticmethod
    def get_instance():
        if Net.model is None:
            Net.model = Net()
            Net.optimizer = torch.optim.Adam(Net.model.parameters(), lr=Net.cfg['lr'], betas=Net.cfg['betas'])
        return Net.model

    def forward(self, data, epoch):
        x, edge_index, y, batch = data.x, data.edge_index, data.y, data.batch
        # x = F.normalize(x, p=1., dim=-1)
        self.x = x

        # x, edge_index, batch = data.x, data.edge_index, data.batch
        e = torch.relu(self.conv1(x, edge_index))
        e = torch.relu(self.conv2(e, edge_index))
        latent = torch.relu(self.conv3(e, edge_index))

        m = torch.relu(self.mask1(latent, edge_index))
        m = torch.relu(self.mask2(m, edge_index))
        softmax = torch.softmax(self.mask3(m, edge_index), dim=1)
        _, max_index = torch.max(softmax, dim=1, keepdim=True)

        pos = torch.arange(0, 2).to(device)
        soft_argmax = torch.sum(softmax * pos, dim=1, keepdim=True)
        self.x_num_nodes = torch.sum(max_index)

        msk_latent = latent * soft_argmax

        c = global_mean_pool(msk_latent, batch)
        c = F.dropout(c, p=0.0, training=self.training)
        c = self.lin1(c)
        c = self.lin2(c)

        graph_indices = batch.bincount()
        current = 0
        r_rate = 0
        self.r_graph = 0
        for gi in graph_indices:
            r_num = torch.sum(max_index[current: current + gi]).float()
            if r_num > 0.0:
                self.r_graph += 1
            r_soft = torch.sum(soft_argmax[current: current + gi]).float()
            r_rate += torch.abs(r_soft / gi.float() - 0.5) * 10
            current += gi

        r_rate /= graph_indices.shape[0]

        # if self.r_graph < graph_indices.shape[0]:
        #     self.penalty = 0
        # else:
        if epoch > 100:
            self.penalty = r_rate - torch.var(soft_argmax)
        else:
            self.penalty = - torch.var(soft_argmax)

        d1 = torch.relu(self.conv4(msk_latent, edge_index))
        d2 = torch.relu(self.conv5(d1, edge_index))
        d3 = self.conv6(d2, edge_index)

        # self.sum_feature = (-torch.var(node_usage) + torch.mean(node_usage)).reshape(1, 1)

        # decode
        return d3, latent * max_index, c

    def train_model(self, train_set, train_set2, valid_set, num_epoch, batch_size):
        model = self.model
        optimizer = self.optimizer
        mse_list = []
        num_nodes_list = []
        total_loss_list = []
        tmp_list = []

        for e in range(num_epoch):
            reconstruction_loss = 0
            nodes_num = 0
            r_graph = 0
            c_loss = 0
            accuracy = 0
            for data in train_set:
                optimizer.zero_grad()
                data = data.to(device)
                z, _, c = model(data, e)
                label = data.y.long()
                print(self.x[:10])
                print(z[:10])

                mse_loss = torch.nn.MSELoss()(z, self.x)
                # c_loss = torch.nn.CrossEntropyLoss()(c, label)
                penalty = self.penalty
                # penalty = 0
                mix_loss = mse_loss + penalty
                mix_loss.backward()

                # pred = c.argmax(dim=1)  # Use the class with highest probability.
                # accuracy += int((pred == label).sum())

                reconstruction_loss += mse_loss
                optimizer.step()
                nodes_num += self.x_num_nodes
                r_graph += self.r_graph

            reconstruction_loss /= len(train_set)

            if e >= 0:
                mse_list.append(reconstruction_loss)
                num_nodes_list.append(nodes_num)
                total_loss_list.append(reconstruction_loss)
                tmp_list.append(0)

            print()
            print('Epoch: {:03d}'.format(e))
            print('AVG Reconstruction Loss:', reconstruction_loss)
            # print('Accuracy:', accuracy)
            # print('C loss:', c_loss)
            print('Nodes num:', nodes_num)
            print('Remaining graph:', r_graph)

        latent_v1 = 0
        for data in train_set2:
            data = data.to(device)
            _, latent_v1, _ = model(data, 0)
            print(self.x_num_nodes, ' ', data.x.shape, ' ', self.r_graph)

        latent_v2 = 0
        for data in valid_set:
            data = data.to(device)
            _, latent_v2, _ = model(data, 0)
            print(self.x_num_nodes, ' ', data.x.shape, ' ', self.r_graph)

        return [mse_list, num_nodes_list, total_loss_list, tmp_list], latent_v1.detach(), latent_v2.detach()

