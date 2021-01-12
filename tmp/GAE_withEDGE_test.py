# %%
import os.path as osp
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool, SAGEConv, GCNConv, GAE
from torch_geometric.utils import train_test_split_edges
from visdom import Visdom
import matplotlib.pyplot as plt

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'MUTAG')
dataset = TUDataset(path, name='MUTAG').shuffle()
test_dataset = dataset[:len(dataset) // 10]
train_dataset = dataset[len(dataset) // 10:]

# Batch(batch=[2281], edge_attr=[5022, 4], edge_index=[2, 5022], x=[2281, 7], y=[128])
channels = 16
N = 0


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        num_features = dataset.num_features
        dim = 32
        self.conv1 = SAGEConv(7, 32)
        self.conv2 = SAGEConv(32, 64)
        self.conv3 = SAGEConv(64, 128)
        self.conv4 = SAGEConv(128, 64)
        self.conv5 = SAGEConv(64, 32)
        self.conv6 = SAGEConv(32, 32)
        self.conv_mu = GCNConv(32, 16, cached=True)
        self.x_num_nodes = 0
        self.sum_feature = 0

    def forward(self, x, edge_index):

        self.sum_feature = 0

        # x, edge_index, batch = data.x, data.edge_index, data.batch      
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = F.relu(self.conv5(x, edge_index))
        x = F.relu(self.conv6(x, edge_index))

        # get used node_num
        x_list = list(x)
        self.x_num_nodes = len(x_list)
        for i in x_list:
            self.sum_feature += torch.sum(i)
            if torch.sum(i) == 0:
                if self.x_num_nodes > 0.33 * len(x_list):
                    self.x_num_nodes -= 1
                else:
                    break

        global N
        N = self.x_num_nodes
        # decode
        return self.conv_mu(x, edge_index)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GAE(Net()).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
crit = torch.nn.MSELoss()
train_loader = DataLoader(train_dataset, batch_size=256)
'''
for data in train_loader:
    data = train_test_split_edges(data)
    break
'''
node_list = None
edge_list = None
for d in train_dataset:
    data = train_test_split_edges(d)
    dx = torch.rand(np.shape(data.x))
    print(dx)
    if node_list is None:
        node_list = dx
    else:
        node_list = torch.cat([node_list, dx], dim=0)
    if edge_list is None:
        edge_list = data.train_pos_edge_index
    else:
        edge_list = torch.cat([edge_list, data.train_pos_edge_index], dim=1)
x, train_pos_edge_index = node_list.to(device), edge_list.to(device)


def trainWithL1AddIn(epoch, lamda, b):
    model.train()  # change mode
    reconstruction_loss = 0
    L1_loss = 0
    loss_all = 0

    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    MSEloss = model.recon_loss(z, train_pos_edge_index) * 0.35
    penalty = model.encoder.sum_feature
    loss = MSEloss + (penalty * lamda + b)
    loss.backward()
    reconstruction_loss += loss.item()
    L1_loss += 0
    loss_all += loss.item()
    optimizer.step()

    return (
        reconstruction_loss, L1_loss, loss_all, 'used nodes: {}'.format(model.encoder.x_num_nodes))


# viz = Visdom()
# assert viz.check_connection()
reconstruction_mse_list = []
l1_penalty_list = []
total_loss_list = []
num_nodes_list = []
lamda = 0.0001
b = 0
num_epoch = 200
for epoch in range(num_epoch):  # run with L1 penalty to node_nums
    loss = trainWithL1AddIn(epoch, lamda, b)
    reconstruction_mse_list.append(loss[0])
    l1_penalty_list.append(loss[1])
    total_loss_list.append(loss[2])
    num_nodes_list.append(N / 3000.0)
    print('  ')
    print('Epoch: {:03d}'.format(epoch))
    print('Loss:', loss[2])
    print('Reconstruction Loss:', loss[0])
    print('L1 Loss:', loss[1])
    print(loss[-1])
plt_x = np.linspace(1, num_epoch, num_epoch)
l1, = plt.plot(plt_x, reconstruction_mse_list, label='Reconstruction_loss')
l2, = plt.plot(plt_x, l1_penalty_list, label='Node_usage_L1_penalty', color='green')
l3, = plt.plot(plt_x, total_loss_list, label='Total_loss', color='red', linestyle='--')
l4, = plt.plot(plt_x, num_nodes_list, label='node number', color='orange')
plt.title('learning_rate={}, epochs={}'.format(1e-1, 200))
plt.ylabel('loss')
plt.xlabel('epoch')
plt.axis([0, num_epoch, 0, 1.5])
plt.legend(handles=[l1, l2, l3, l4, ], labels=['Reconstruction_loss', 'Node_usage_L1_penalty', 'Total_loss', 'remaining node number'], loc='best')
# plt.legend(handles=[l1], labels=['Reconstruction_loss'], loc='best')

# viz.matplot(plt)
plt.show()
