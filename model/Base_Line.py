from torch_geometric.utils import sort_edge_index
from torch_geometric.nn import SAGEConv as conv
import torch.nn.functional as F
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(torch.nn.Module):
    model = None
    optimizer = None

    cfg = {'lr': 2e-4,
           'betas': (0.9, 0.999),
           'lbd': 1,
           'thresh': 0.6}

    def __init__(self):
        super(Net, self).__init__()

        hidden = 128
        dropout = 0.5
        # encoder
        self.conv1 = conv(500, hidden)
        self.conv2 = conv(hidden, hidden)
        self.conv3 = conv(hidden, hidden)
        # self.conv4 = conv(hidden, hidden // 2)
        # self.conv5 = conv(hidden // 2, hidden // 2)
        # self.conv6 = conv(hidden // 2, hidden // 2)
        # self.conv7 = conv(hidden // 2, hidden // 4)
        # self.conv8 = conv(hidden // 4, hidden // 4)
        # self.conv9 = conv(hidden // 4, hidden // 4)
        # # decoder
        # self.conv12 = conv(hidden // 4, hidden // 4)
        # self.conv13 = conv(hidden // 4, hidden // 4)
        # self.conv14 = conv(hidden // 4, hidden // 2)
        # self.conv15 = conv(hidden // 2, hidden // 2)
        # self.conv16 = conv(hidden // 2, hidden // 2)
        # self.conv17 = conv(hidden // 2, hidden)
        self.conv18 = conv(hidden, hidden)
        self.conv19 = conv(hidden, hidden)
        self.conv20 = conv(hidden, 500)

    @staticmethod
    def get_instance():
        if Net.model is None:
            Net.model = Net()
            Net.optimizer = torch.optim.Adam(Net.model.parameters(), lr=Net.cfg['lr'], betas=Net.cfg['betas'])
        return Net.model

    def forward(self, data):

        x, edge_index = data.x, data.edge_index
        self.original_x = x
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # x, edge_index, batch = data.x, data.edge_index, data.batch
        e = F.relu(self.conv1(x, edge_index))
        e = F.relu(self.conv2(e, edge_index))
        e = F.relu(self.conv3(e, edge_index))
        # e = F.relu(self.conv4(e, edge_index))
        # e = F.relu(self.conv5(e, edge_index))
        # e = F.relu(self.conv6(e, edge_index))
        # e = F.relu(self.conv7(e, edge_index))
        # e = F.relu(self.conv8(e, edge_index))
        # e = F.relu(self.conv9(e, edge_index))
        # e = F.relu(self.conv12(e, edge_index))
        # e = F.relu(self.conv13(e, edge_index))
        # e = F.relu(self.conv14(e, edge_index))
        # e = F.relu(self.conv15(e, edge_index))
        # e = F.relu(self.conv16(e, edge_index))
        # e = F.relu(self.conv17(e, edge_index))
        e = F.relu(self.conv18(e, edge_index))
        e = F.relu(self.conv19(e, edge_index))
        e = F.relu(self.conv20(e, edge_index))

        node_usage = 0
        self.sum_feature = 0
        self.x_num_nodes = 0
        # decode
        return e

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
                print(z[:5], label[:5])

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
            for data in valid_set:
                optimizer.zero_grad()
                data = data.to(device)
                z = model(data)
                label = self.original_x

                mse_loss = torch.nn.MSELoss()(z, label)
                valid_set_nodes += self.x_num_nodes

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
