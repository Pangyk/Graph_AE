from torch_geometric.data import DataLoader
from utils.CustomDataSet import SceneGraphs
from model.GraphUNet import Net
import utils.Display_Plot as dp
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epoch = 1000
batch_size = 1000
model = Net.get_instance().to(device)

# SelectGraph.data_name = 'COLORS-3'
# SelectGraph.thresh = 8
# SelectGraph.direction = -1
# data_set1 = SelectGraph('Data/MUTAG/Train/')
# SelectGraph.direction = 1
# data_set2 = SelectGraph('Data/MUTAG/Test/')
# train_set = DataLoader(data_set1[:5000], batch_size, shuffle=False)
# valid_set = DataLoader(data_set1[1500:2000], batch_size, shuffle=False)

data_set1 = SceneGraphs('data/scene/')
train_set = DataLoader(data_set1[2000:4000], batch_size=500, shuffle=False)
train_set2 = DataLoader(data_set1[:100], batch_size, shuffle=False)
valid_set = DataLoader(data_set1[:100], batch_size, shuffle=False)

data_list = model.train_model(train_set, valid_set, num_epoch)

title = "learning rate: 1e-4, epochs: 500"
labels = ['MSE Loss', 'Penalty Loss', 'Total Loss', 'Used Nodes']
dp.display_one(data_list, num_epoch, labels, title)
dp.display(data_list, num_epoch, labels, title)

