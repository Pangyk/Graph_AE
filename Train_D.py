from torch_geometric.data import DataLoader
from utils.CustomDataSet import ExistGraph
from model.Node_Comp_Mask import Net
import utils.Display_Plot as dp
import torch

num_epoch = 500
batch_size = 1000
file_dir_1 = "data/random/"
file_dir_2 = "data/Pubmed-Diabetes/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net.get_instance().to(device)
data_set = ExistGraph.get_data_set()
train_set = DataLoader(data_set[3:], batch_size, shuffle=True)
valid_set = DataLoader(data_set[:3], batch_size=1, shuffle=False)
data_list = model.train_model(train_set, valid_set, num_epoch)

title = "learning rate: 1e-4, epochs: 500"
labels = ['MSE Loss', 'Penalty Loss', 'Total Loss', 'Used Nodes']
dp.display_one(data_list, num_epoch, labels, title)
dp.display(data_list, num_epoch, labels, title)
