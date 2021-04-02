from abc import ABC

from torch_geometric.nn import global_mean_pool, SAGEConv
import torch.nn.functional as F
import torch

device = torch.device('cuda:1')


class MLP(torch.nn.Module, ABC):
    model = None
    optimizer = None

    cfg = {'lr': 1e-2}

    def __init__(self):
        super(MLP, self).__init__()
        ipt_size = 64
        hidden = 200

        self.conv1 = SAGEConv(64, 64)
        self.conv2 = SAGEConv(64, 64)
        self.conv3 = SAGEConv(64, 64)

        self.lin1 = torch.nn.Linear(ipt_size, hidden)
        self.bn1 = torch.nn.BatchNorm1d(hidden)
        self.lin2 = torch.nn.Linear(hidden, hidden // 2)
        self.bn2 = torch.nn.BatchNorm1d(hidden // 2)
        self.lin3 = torch.nn.Linear(hidden // 2, hidden // 2)
        self.bn3 = torch.nn.BatchNorm1d(hidden // 2)
        self.lin4 = torch.nn.Linear(hidden // 2, 80)

    @staticmethod
    def get_instance():
        if MLP.model is None:
            MLP.model = MLP()
            MLP.optimizer = torch.optim.Adam(MLP.model.parameters(), lr=MLP.cfg['lr'], weight_decay=0.001)
        return MLP.model

    def forward(self, x, edge_index, batch):
        x1 = self.conv1(x, edge_index)
        x1 = self.conv2(x1, edge_index)
        x1 = self.conv3(x1, edge_index)
        c = global_mean_pool(x1, batch)
        c = F.dropout(c, p=0.1, training=self.training)
        c = self.lin1(c)
        b = self.bn1(c)
        b = torch.tanh(b)
        b = F.dropout(b, p=0.1, training=self.training)
        c = self.lin2(b)
        b = self.bn2(c)
        b = torch.tanh(b)
        b = F.dropout(b, p=0.1, training=self.training)
        c = self.lin3(b)
        b = self.bn3(c)
        b = torch.tanh(b)
        c = self.lin4(b)

        return c

    def train_model(self, train_set, valid_set, num_epoch, group1, group2):
        model = self.model
        optimizer = self.optimizer
        t_c_list = []
        t_accuracy_list = []
        c_list = []
        accuracy_list = []

        for e in range(num_epoch):
            t_c_loss = 0
            c_loss = 0
            accuracy = 0
            total_num = 0
            for data in train_set:
                optimizer.zero_grad()
                data = data.to(device)
                self.training = True
                c = model(group1[0], group1[1], group1[2])
                label = data.y.long()

                pred = c.argmax(dim=1)
                total_num += label.shape[0]
                accuracy += int((pred == label).sum())
                regularization_loss = 0
                # for param in model.parameters():
                #     regularization_loss += torch.sum(abs(param))

                c_loss = torch.nn.CrossEntropyLoss()(c, label) + 1e-3 * regularization_loss
                t_c_loss = c_loss
                c_loss.backward()
                optimizer.step()

            accuracy /= total_num
            t_c_list.append(t_c_loss)
            t_accuracy_list.append(accuracy)
            print()
            print('Epoch: {:03d}'.format(e))
            print('AVG c Loss:', c_loss)
            print('Train Accuracy:', accuracy)

            accuracy = 0
            total_num = 0
            for data in valid_set:
                data = data.to(device)
                self.training = False
                c = model(group2[0], group1[1], group2[2])
                pred = c.argmax(dim=1)
                label = data.y.long()
                total_num += label.shape[0]
                accuracy += int((pred == label).sum())

            accuracy /= total_num

            print('Test Accuracy:', accuracy)

            c_list.append(0)
            accuracy_list.append(accuracy)

        return [t_c_list, t_accuracy_list, c_list, accuracy_list]
