from torch_geometric.nn import SAGEConv, GATConv, GCNConv
import utils.Display_Statistic as ds
import torch.nn.functional as F
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(torch.nn.Module):
    model = None
    optimizer = None

    cfg = {'lr': 1e-4,
           'betas': (0.9, 0.999),
           'lbd': 0.5,
           'thresh': torch.tensor(0.2).to(device),
           'p1': torch.tensor([10.0]).to(device),
           'p0': torch.tensor([100.0]).to(device)}

    def __init__(self):
        super(Net, self).__init__()

        # encoder
        self.conv1 = GATConv(42, 128)
        self.conv2 = GATConv(128, 128)
        self.conv3 = GATConv(128, 128)
        self.conv4 = GATConv(128, 128)
        # mask
        self.mask1 = torch.nn.Linear(128, 64)
        self.mask2 = torch.nn.Linear(64, 16)
        self.mask3 = torch.nn.Linear(16, 16)
        self.mask4 = torch.nn.Linear(16, 2)
        # decoder
        self.conv5 = GATConv(128, 128)
        self.conv6 = GATConv(128, 128)
        self.conv7 = GATConv(128, 128)
        self.conv8 = GATConv(128, 42)

        self.x_num_nodes = 0
        self.r_rate = 0
        self.sum_feature = 0

    @staticmethod
    def get_instance():
        if Net.model is None:
            Net.model = Net()
            Net.optimizer = torch.optim.Adam(Net.model.parameters(), lr=Net.cfg['lr'], betas=Net.cfg['betas'])
        return Net.model

    def forward(self, data, epoch):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.normalize(x, p=1., dim=-1)

        # x, edge_index, batch = data.x, data.edge_index, data.batch
        e = torch.tanh(self.conv1(x, edge_index))
        e = torch.tanh(self.conv2(e, edge_index))
        e = torch.tanh(self.conv3(e, edge_index))
        latent = torch.tanh(self.conv4(e, edge_index))

        m1 = torch.tanh(self.mask1(latent))
        m2 = torch.tanh(self.mask2(m1))
        m3 = torch.tanh(self.mask3(m2))
        m4 = self.mask4(m3)
        softmax = torch.softmax(m4, dim=1)
        _, max_index = torch.max(softmax, dim=1, keepdim=True)

        pos = torch.arange(0, 2).to(device)
        soft_argmax = torch.sum(softmax * pos, dim=1, keepdim=True)
        msk_latent = latent * soft_argmax
        self.x_num_nodes = torch.sum(max_index)
        if epoch > 100:
            self.penalty = torch.abs(torch.mean(soft_argmax) - 0.8) \
                           - torch.var(soft_argmax)
        else:
            self.penalty = - torch.var(soft_argmax)

        d = torch.tanh(self.conv5(msk_latent, edge_index))
        d = torch.tanh(self.conv6(d, edge_index))
        d = torch.tanh(self.conv7(d, edge_index))
        d = torch.tanh(self.conv8(d, edge_index))

        # self.sum_feature = (-torch.var(node_usage) + torch.mean(node_usage)).reshape(1, 1)

        self.bin_batch = batch.bincount().int()
        self.max_index = max_index
        self.latent = latent
        self.batch = batch

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
                z = model(data, e)
                label = self.original_x

                mse_loss = 2 * torch.nn.MSELoss()(z, label)
                penalty = self.penalty
                # penalty = 0
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
                z = model(data, e)
                label = self.original_x

                mse_loss = 2 * torch.nn.MSELoss()(z, label)
                valid_set_nodes += self.x_num_nodes

                if e == 0:
                    print(data.y)

                if e == num_epoch - 1:
                    print(self.max_index)
                    print(data.edge_index)

            reconstruction_loss /= len(train_set)
            penalty_loss /= len(train_set)
            total_loss /= len(train_set)
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

            # if (e + 1) % 20 == 0:
            # ds.print_feature(self.d, self.original_x)
            # print(self.latent[0])

        return [mse_list, penalty_list, total_loss_list, num_nodes_list]
