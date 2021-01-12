from graph_ae.UNet import GraphUNet
import torch.nn.functional as F
import torch
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(torch.nn.Module):
    model = None
    optimizer = None

    cfg = {'lr': 1e-2,
           'betas': (0.9, 0.999)}

    def __init__(self):
        super(Net, self).__init__()

        self.conv = GraphUNet(20, 64, 20, 3, 0.8, True, act=F.relu).to(device)
        self.original_x = None
        self.sum_feature = 0
        self.x_num_nodes = 0

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
        self.original_x = x

        d = self.conv(x, edge_index, batch).to(device)

        return d

    def train_model(self, train_set, valid_set, num_epoch):
        model = self.model
        optimizer = self.optimizer
        mse_list = []
        penalty_list = []
        total_loss_list = []
        num_nodes_list = []

        for e in range(num_epoch):
            reconstruction_loss = 0
            penalty_loss = 0
            total_loss = 0
            nodes_num = 0

            for data in train_set:
                optimizer.zero_grad()
                data = data.to(device)
                z = model(data)
                label = self.original_x

                mse_loss = torch.nn.MSELoss()(z, label)
                penalty = 0
                mix_loss = mse_loss + penalty
                mix_loss.backward()

                reconstruction_loss += mse_loss
                penalty_loss += penalty
                total_loss += mix_loss.item()
                optimizer.step()
                nodes_num += self.x_num_nodes

            valid_set_nodes = 0
            mse_loss = 0
            # for data in valid_set:
            #     data = data.to(device)
            #     z = model(data)
            #     label = self.original_x
            #
            #     mse_loss = torch.nn.MSELoss()(z, label)
            #     valid_set_nodes += self.x_num_nodes

            reconstruction_loss /= len(train_set)
            penalty_loss /= len(train_set)
            total_loss /= len(train_set)
            mse_list.append(reconstruction_loss)
            penalty_list.append(penalty_loss)
            total_loss_list.append(total_loss)
            num_nodes_list.append(nodes_num)

            print()
            print('Epoch: {:03d}'.format(e))
            print('total Loss:', total_loss)
            print('Reconstruction Loss:', reconstruction_loss)
            print('Penalty Loss:', penalty_loss)
            print('nodes num:', nodes_num)
            print("======================")
            print("Validation MSE", mse_loss)
            print("Validation node usage", valid_set_nodes)

        return [mse_list, penalty_list, total_loss_list, num_nodes_list]
