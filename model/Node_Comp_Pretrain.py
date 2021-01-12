from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(torch.nn.Module):
    model = None
    optimizer = None

    cfg = {'lr': 1e-4,
           'betas': (0.9, 0.95),
           'lbd': 1,
           'thresh': 0.2}

    def __init__(self):
        super(Net, self).__init__()

        # encoder
        self.conv1 = SAGEConv(500, 256, normalize=False)
        self.conv2 = SAGEConv(256, 128, normalize=False)
        self.conv3 = SAGEConv(128, 128, normalize=False)
        self.conv4 = SAGEConv(128, 128, normalize=True)
        # decoder
        self.conv5 = SAGEConv(128, 128, normalize=False)
        self.conv6 = SAGEConv(128, 128, normalize=False)
        self.conv7 = SAGEConv(128, 256, normalize=False)
        self.conv8 = SAGEConv(256, 500, normalize=False)

    @staticmethod
    def get_instance():
        if Net.model is None:
            Net.model = Net()
            Net.optimizer = torch.optim.Adam(Net.model.parameters(), lr=Net.cfg['lr'], betas=Net.cfg['betas'])
        return Net.model

    def forward(self, data):

        x, edge_index = data.x, data.edge_index
        self.original_x = x

        # x, edge_index, batch = data.x, data.edge_index, data.batch
        e = F.relu(self.conv1(x, edge_index))
        e = F.relu(self.conv2(e, edge_index))
        e = F.relu(self.conv3(e, edge_index))
        latent = F.relu(self.conv4(e, edge_index))
        d = F.relu(self.conv5(latent, edge_index))
        d = F.relu(self.conv6(d, edge_index))
        d = F.relu(self.conv7(d, edge_index))
        d = F.relu(self.conv8(d, edge_index))

        node_usage = torch.sum(latent, dim=1)
        self.sum_feature = torch.sum(node_usage).reshape(1, 1)
        self.x_num_nodes = (node_usage > 0).sum(dim=0)
        # decode
        return d

    def train_model(self, data_set, num_epoch):
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

            for data in data_set:
                optimizer.zero_grad()
                z = model(data.to(device))
                label = self.original_x

                mse_loss = 2 * torch.nn.MSELoss()(z, label) * 1000

                if e <= 100:
                    penalty = 0
                else:
                    sum_f = self.sum_feature * Net.cfg['lbd'] - Net.cfg['thresh']
                    penalty = torch.clamp(sum_f, min=0.0)

                mix_loss = mse_loss + penalty
                mix_loss.backward()

                reconstruction_loss = mse_loss
                penalty_loss = penalty
                total_loss = mix_loss.item()
                optimizer.step()
                nodes_num += self.x_num_nodes

            mse_list.append(reconstruction_loss)
            penalty_list.append(penalty_loss)
            total_loss_list.append(total_loss)
            num_nodes_list.append(nodes_num)

            print()
            print('Epoch: {:03d}'.format(e))
            print('AVG total Loss:', total_loss)
            print('AVG Reconstruction Loss:', reconstruction_loss)
            print('AVG Penalty Loss:', penalty_loss)
            print('nodes num:', nodes_num)

        return [mse_list, penalty_list, total_loss_list, num_nodes_list]
