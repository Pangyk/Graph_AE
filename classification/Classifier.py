from abc import ABC

from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLP(torch.nn.Module, ABC):
    model = None
    optimizer = None

    cfg = {'lr': 1e-2,
           'betas': (0.9, 0.999)}

    def __init__(self):
        super(MLP, self).__init__()
        hidden = 32
        self.lin1 = torch.nn.Linear(hidden, hidden)
        self.lin2 = torch.nn.Linear(hidden, hidden)
        self.lin3 = torch.nn.Linear(hidden, 11)

    @staticmethod
    def get_instance():
        if MLP.model is None:
            MLP.model = MLP()
            MLP.optimizer = torch.optim.Adam(MLP.model.parameters(), lr=MLP.cfg['lr'], betas=MLP.cfg['betas'])
        return MLP.model

    def forward(self, x, batch):
        c = global_mean_pool(x, batch)
        c = F.dropout(c, p=0.0, training=self.training)
        c = torch.relu(self.lin1(c))
        c = torch.relu(self.lin2(c))
        c = self.lin3(c)

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
                c = model(group1[0], group1[2])
                label = data.y.long()

                pred = c.argmax(dim=1)
                total_num += label.shape[0]
                accuracy += int((pred == label).sum())

                c_loss = torch.nn.CrossEntropyLoss()(c, label)
                t_c_loss = c_loss
                c_loss.backward()
                optimizer.step()

            accuracy /= total_num
            if e >= 200:
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
                c = model(group2[0], group2[2])
                pred = c.argmax(dim=1)
                label = data.y.long()
                total_num += label.shape[0]
                accuracy += int((pred == label).sum())

            accuracy /= total_num

            print('Test Accuracy:', accuracy)

            if e >= 200:
                c_list.append(0)
                accuracy_list.append(accuracy)

        return [t_c_list, t_accuracy_list, c_list, accuracy_list]
