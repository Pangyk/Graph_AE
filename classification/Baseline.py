from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch.nn import Linear
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(torch.nn.Module):
    model = None
    optimizer = None

    cfg = {'lr': 1e-2,
           'betas': (0.9, 0.999),
           'lbd': 1,
           'thresh': 0.6}

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = GCNConv(5, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)

        self.lin1 = torch.nn.Linear(64, 32)
        self.lin2 = torch.nn.Linear(32, 32)
        self.lin3 = torch.nn.Linear(32, 32)
        self.lin4 = torch.nn.Linear(32, 11)

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
        e = e.tanh()
        e = self.conv2(e, edge_index)
        e = e.tanh()
        e = self.conv3(e, edge_index)

        # 2. Readout layer
        p = global_mean_pool(e, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        d = F.dropout(p, p=0.0, training=self.training)
        c = torch.relu(self.lin1(d))
        c = torch.relu(self.lin2(c))
        c = torch.relu(self.lin3(c))
        c = self.lin4(c)

        return c

    def train_model(self, train_set, valid_set, num_epoch, batch_size):
        model = self.model
        optimizer = self.optimizer
        t_c_list = []
        penalty_list = []
        t_accuracy_list = []
        accuracy_list = []

        for e in range(num_epoch):
            reconstruction_loss = 0
            accuracy = 0
            for data in train_set:
                optimizer.zero_grad()
                data = data.to(device)
                c = model(data)
                label = self.label

                pred = c.argmax(dim=1)
                accuracy = int((pred == label).sum())

                c_loss = torch.nn.CrossEntropyLoss()(c, label)
                c_loss.backward()

                reconstruction_loss += c_loss
                optimizer.step()

            accuracy /= 100.0
            t_c_list.append(reconstruction_loss)
            t_accuracy_list.append(accuracy)
            print()
            print('Epoch: {:03d}'.format(e))
            print('Reconstruction Loss:', reconstruction_loss)
            print("======================")
            print("Accuracy: ", accuracy)

            accuracy = 0
            for data in valid_set:
                data = data.to(device)
                c = model(data)
                label = self.label

                pred = c.argmax(dim=1)  # Use the class with highest probability.
                accuracy = int((pred == label).sum())

            accuracy /= batch_size
            penalty_list.append(0)
            accuracy_list.append(accuracy)

            print('Test Accuracy:', accuracy)

        return [t_c_list, t_accuracy_list, penalty_list, accuracy_list]
