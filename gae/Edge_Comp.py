from torch_geometric.nn import SAGEConv, GCNConv, VGAE
import utils.Display_Statistic as ds
import torch.nn.functional as F
import numpy as np
from scipy.sparse import coo_matrix
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(torch.nn.Module):
    model = None
    instance = None
    optimizer = None

    cfg = {'lr': 1e-2,
           'betas': (0.9, 0.999),
           'lbd': 0.5,
           'thresh': torch.tensor(0.2).to(device),
           'p1': torch.tensor([10.0]).to(device),
           'p0': torch.tensor([100.0]).to(device)}

    def __init__(self):
        super(Net, self).__init__()

        # encoder
        self.conv1 = GCNConv(500, 32, cached=True)
        self.mu = GCNConv(32, 32, cached=True)
        self.logstd = GCNConv(32, 32, cached=True)

    @staticmethod
    def get_instance():
        if Net.model is None:
            Net.instance = Net()
            Net.model = VGAE(Net.instance).to(device)
            Net.optimizer = torch.optim.Adam(Net.model.parameters(), lr=Net.cfg['lr'], betas=Net.cfg['betas'])
        return Net.instance

    def forward(self, x, edge_index):

        # x, edge_index, batch = data.x, data.edge_index, data.batch
        e = torch.tanh(self.conv1(x, edge_index))
        mu = self.mu(e, edge_index)
        logstd = self.logstd(e, edge_index)

        return mu, logstd

    def train_model(self, train_set, valid_set, original_e, num_epoch):
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
                model.train()
                optimizer.zero_grad()
                data = data.to(device)
                z = model.encode(data.x, data.train_pos_edge_index)
                # print(torch.sum(z))
                loss = model.recon_loss(z, data.train_pos_edge_index)
                loss = loss + (1 / data.x.shape[0]) * model.kl_loss()
                loss.backward()
                reconstruction_loss = loss.item()
                optimizer.step()

            for data in valid_set:
                model.eval()
                optimizer.zero_grad()
                data = data.to(device)
                z = model.encode(data.x, data.train_pos_edge_index)

                auc, ap = model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)
                print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(e, auc, ap))
                if e == 99:
                    e_predict = torch.sigmoid(torch.matmul(z, z.t()))
                    es = e_predict.detach().cpu().numpy()
                    edges = original_e.detach().cpu().numpy()
                    label = coo_matrix((np.ones(np.shape(edges)[1]), (edges[0], edges[1])), shape=(np.shape(es)),
                                       dtype=np.float32)
                    predict = np.float32(es > 0.95)
                    print(np.sum(predict))
                    result = label - predict
                    print("True negative: ", np.sum(np.float32(result > 0)))
                    print("False positive: ", np.sum(np.float32(result < 0)))
                    print("Other: ", np.sum(np.float32(result == 0)))
                    print("Real edges: ", np.shape(edges)[1])
                    print("Num nodes: ", np.shape(es)[0])

            mse_list.append(reconstruction_loss)
            penalty_list.append(penalty_loss)
            total_loss_list.append(total_loss)
            num_nodes_list.append(nodes_num)

            # print()
            # print('Epoch: {:03d}'.format(e))
            # print('AVG Reconstruction Loss:', reconstruction_loss)
            # print("======================")
            # print("Validation MSE", mse_loss)

        return [mse_list, penalty_list, total_loss_list, num_nodes_list]
