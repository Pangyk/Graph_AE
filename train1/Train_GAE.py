from torch_geometric.data import DataLoader
from utils.CustomDataSet import SelectGraph, SceneGraphs
from classification.Graph_AE import Net
from classification.Classifier import MLP
import utils.Display_Plot as dp
import torch

device = torch.device('cuda')

num_epoch = 500
batch_size = 200
comp_model = Net.get_instance().to(device)
cfy_model = MLP.get_instance().to(device)


SelectGraph.data_name = 'Shana7000'
data_set_Shana = SelectGraph('data/' + SelectGraph.data_name)
train_set = DataLoader(data_set_Shana[:5000], batch_size=500, shuffle=True)
train_set2 = DataLoader(data_set_Shana[5000:6000], 1000, shuffle=False)
test_set = DataLoader(data_set_Shana[6000:7000], 1000, shuffle=False)

data_list1, group1, group2 = comp_model.train_model(train_set, train_set2, test_set, num_epoch)
data_list2 = cfy_model.train_model(train_set2, test_set, int(num_epoch // 4), group1, group2)

title = "GAE Normal"
labels = ['MSE Loss', 'Num Nodes', 'Total Loss', title]
dp.display(data_list1, num_epoch, labels, title)
labels = ['Train Loss', 'Train Acc', title, 'Test Acc']
dp.display(data_list2, int(num_epoch // 4), labels, title)
