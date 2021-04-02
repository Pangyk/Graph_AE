from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch.nn import Linear
import torch

device = torch.device('cuda:1')


class Net(torch.nn.Module):
    model = None
    optimizer = None

    cfg = {'lr': 1e-2,
           'betas': (0.9, 0.999),
           'lbd': 1,
           'thresh': 0.6}

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = SAGEConv(500, 64)
        self.conv2 = SAGEConv(64, 64)
        self.conv3 = SAGEConv(64, 64)

        ipt_size = 64
        hidden = 200
        self.lin1 = torch.nn.Linear(ipt_size, hidden)
        self.bn1 = torch.nn.BatchNorm1d(hidden)
        self.lin2 = torch.nn.Linear(hidden, hidden // 2)
        self.bn2 = torch.nn.BatchNorm1d(hidden // 2)
        self.lin3 = torch.nn.Linear(hidden // 2, hidden // 2)
        self.bn3 = torch.nn.BatchNorm1d(hidden // 2)
        self.lin4 = torch.nn.Linear(hidden // 2, 80)

    @staticmethod
    def get_instance():
        if Net.model is None:
            Net.model = Net()
            Net.optimizer = torch.optim.Adam(Net.model.parameters(), lr=Net.cfg['lr'], betas=Net.cfg['betas'])
        return Net.model

    def forward(self, data):
        x, edge_index, y, batch = data.x, data.edge_index, data.y, data.batch
        x = F.normalize(x, p=1., dim=-1)
        self.original_x = x
        self.label = y.long()

        e = self.conv1(x, edge_index)
        e = self.conv2(e, edge_index)
        e = self.conv3(e, edge_index)

        # 2. Readout layer
        c = global_mean_pool(e, batch)
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

    def train_model(self, train_set, valid_set, num_epoch, batch_size):
        model = self.model
        optimizer = self.optimizer
        t_c_list = []
        penalty_list = []
        t_accuracy_list = []
        accuracy_list = []

        # mean, std = None, None
        for e in range(num_epoch):
            reconstruction_loss = 0
            accuracy = 0
            total_case = 0
            for data in train_set:
                optimizer.zero_grad()
                data = data.to(device)
                self.training = True
                # if mean is None and std is None:
                #     x = data.x
                #     mean = torch.mean(x, dim=0)
                #     std = torch.std(x, dim=0) + 1e-12
                # data.x = (data.x - mean) / std
                c = model(data)
                label = self.label

                pred = c.argmax(dim=1)
                accuracy = int((pred == label).sum())
                total_case += label.shape[0]

                c_loss = torch.nn.CrossEntropyLoss()(c, label)
                c_loss.backward()

                reconstruction_loss += c_loss
                optimizer.step()

            accuracy /= total_case
            t_c_list.append(reconstruction_loss)
            t_accuracy_list.append(accuracy)
            print()
            print('Epoch: {:03d}'.format(e))
            print('Reconstruction Loss:', reconstruction_loss)
            print("======================")
            print("Accuracy: ", accuracy)

            accuracy = 0
            total_case = 0
            for data in valid_set:
                data = data.to(device)
                # data.x = (data.x - mean) / std
                self.training = False
                c = model(data)
                label = self.label
                total_case += label.shape[0]
                pred = c.argmax(dim=1)  # Use the class with highest probability.
                accuracy = int((pred == label).sum())

            accuracy /= total_case
            penalty_list.append(0)
            accuracy_list.append(accuracy)

            print('Test Accuracy:', accuracy)

        return [t_c_list, t_accuracy_list, penalty_list, accuracy_list]
