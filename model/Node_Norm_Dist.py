from torch_geometric.nn import SAGEConv
import utils.Display_Statistic as ds
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(torch.nn.Module):
    model = None
    optimizer = None

    cfg = {'lr': 1e-4,
           'betas': (0.9, 0.99),
           'lbd': 0.5,
           'thresh': 0.0}

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

        x, edge_index, batch = data.x, data.edge_index, data.batch

        # x, edge_index, batch = data.x, data.edge_index, data.batch
        e = torch.relu(self.conv1(x, edge_index))
        e = torch.relu(self.conv2(e, edge_index))
        e = torch.relu(self.conv3(e, edge_index))
        latent = torch.relu(self.conv4(e, edge_index))
        d = torch.relu(self.conv5(latent, edge_index))
        d = torch.relu(self.conv6(d, edge_index))
        d = torch.relu(self.conv7(d, edge_index))
        d = torch.relu(self.conv8(d, edge_index))

        node_usage = torch.sum(latent, dim=1)

        self.x_num_nodes = (node_usage > 0).sum(dim=0)

        # self.sum_feature = (-torch.var(node_usage) + torch.mean(node_usage)).reshape(1, 1)
        self.sum_feature = (-torch.var(node_usage)).reshape(1, 1)

        self.node_usage = node_usage
        self.latent = latent
        self.batch = batch
        self.bin_batch = batch.bincount().int()
        self.original_x = x
        self.original_e = edge_index
        self.d = d

        # decode
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

                mse_loss = 2 * torch.nn.MSELoss()(z, label)
                # penalty = self.sum_feature * Net.cfg['lbd'] - Net.cfg['thresh']
                penalty = 0
                mix_loss = mse_loss + penalty
                mix_loss.backward()

                reconstruction_loss = mse_loss
                penalty_loss = penalty
                total_loss = mix_loss.item()
                optimizer.step()
                nodes_num += self.x_num_nodes

            valid_set_nodes = 0
            mse_loss = 0
            for data in valid_set:
                optimizer.zero_grad()
                data = data.to(device)
                z = model(data)
                label = self.original_x

                mse_loss = 2 * torch.nn.MSELoss()(z, label)
                valid_set_nodes += self.x_num_nodes

            mse_list.append(reconstruction_loss)
            penalty_list.append(penalty_loss)
            total_loss_list.append(total_loss)
            num_nodes_list.append(nodes_num)

            print()
            print('Epoch: {:03d}'.format(e))
            print('AVG total Loss:', total_loss)
            print('AVG Reconstruction Loss:', reconstruction_loss)
            print('AVG Penalty Loss:', penalty_loss)
            print('Nodes num:', nodes_num)
            print("======================")
            print("Validation MSE", mse_loss)
            print("Validation node usage", valid_set_nodes)

            if (e + 1) % 20 == 0:
                # ds.print_feature(self.d, self.original_x)
                print(self.latent[0])

        return [mse_list, penalty_list, total_loss_list, num_nodes_list]
