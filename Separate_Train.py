from torch_geometric.data import DataLoader
from utils.CustomDataSet import SelectGraph
from classification.Comp_Cfy import Net
from classification.Classifier import MLP
import utils.Display_Plot as dp
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epoch = 500
batch_size = 200
comp_model = Net.get_instance().to(device)
cfy_model = MLP.get_instance().to(device)

SelectGraph.data_name = 'COLORS-3'
SelectGraph.thresh = 8
SelectGraph.direction = -1
data_set1 = SelectGraph('Data/MUTAG/Train/')
SelectGraph.direction = 1
data_set2 = SelectGraph('Data/MUTAG/Test/')
train_set = DataLoader(data_set1[:5000], batch_size=1000, shuffle=True)
train_set2 = DataLoader(data_set2[:100], batch_size, shuffle=False)
valid_set = DataLoader(data_set2[100:300], batch_size, shuffle=False)

data_list1, latent_v1, latent_v2 = comp_model.train_model(train_set, train_set2, valid_set, num_epoch, batch_size)
# data_list2 = cfy_model.train_model(train_set2, valid_set, num_epoch, batch_size, latent_v1, latent_v2)
data_list = data_list1

title = "learning rate: 1e-4, epochs: 500"
labels = ['MSE Loss', 'Num nodes', 'Total Loss', '']
dp.display_one(data_list, num_epoch, labels, title)
dp.display(data_list, num_epoch, labels, title)
