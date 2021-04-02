from torch_geometric.data import DataLoader
from utils.CustomDataSet import SceneGraphs, SelectGraph
from classification.Baseline import Net
import utils.Display_Plot as dp
import torch
import os


batch_size = 100
num_epoch = 200
device = torch.device('cuda:1')

model = Net.get_instance().to(device)

SelectGraph.data_name = 'Shana7000'
data_set_Shana = SelectGraph(os.path.join('data', SelectGraph.data_name))
train_set = DataLoader(data_set_Shana[5000:5500], 500, shuffle=False)
test_set = DataLoader(data_set_Shana[6000:7000], 1000, shuffle=False)

data_list = model.train_model(train_set, test_set, num_epoch, batch_size)

title = "learning rate: 1e-2, epochs: 100"
labels = ['MSE Loss', 'Penalty Loss', 'Total Loss', 'Used Nodes']
dp.display_one(data_list, num_epoch, labels, title)
dp.display(data_list, num_epoch, labels, title)

