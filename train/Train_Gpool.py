from torch_geometric.data import DataLoader
from utils.CustomDataSet import SelectGraph, SceneGraphs
from classification.Gpool_model import Net
from classification.Classifier import MLP
import utils.Display_Plot as dp
import torch

device = torch.device('cuda:1')

num_epoch = 100
batch_size = 200
comp_model = Net.get_instance().to(device)
cfy_model = MLP.get_instance().to(device)

SelectGraph.data_name = 'Shana7000'
data_set_Shana = SelectGraph('data/' + SelectGraph.data_name)
train_set = DataLoader(data_set_Shana[:5000], 500, shuffle=True)

SelectGraph.data_name = 'Shana7000'
data_set_Shana = SelectGraph('data/' + SelectGraph.data_name)
train_set2 = DataLoader(data_set_Shana[5000:6000], 1000, shuffle=False)
test_set = DataLoader(data_set_Shana[6000:7000], 1000, shuffle=False)

m_name = "Gpool_TRANSFER.ckpt"
data_list1, group1, group2 = comp_model.train_model(train_set, train_set2, test_set, num_epoch, m_name)
data_list2 = cfy_model.train_model(train_set2, test_set, int(num_epoch // 2), group1, group2)

title = "SAG TRANSFER"
labels = ['MSE Loss', 'Num Nodes', 'Total Loss', title]
dp.display(data_list1, num_epoch, labels, title)
labels = ['Train Loss', 'Train Acc', title, 'Test Acc']
dp.display(data_list2, int(num_epoch // 2), labels, title)
