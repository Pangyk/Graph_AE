from torch_geometric.data import DataLoader
from utils.CustomDataSet import SelectGraph
from classification.Baseline import Net
import utils.Display_Plot as dp
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epoch = 400
batch_size = 200
comp_model = Net.get_instance().to(device)

SelectGraph.data_name = 'COLORS-3'
SelectGraph.thresh = 8
SelectGraph.direction = 1
data_set = SelectGraph('Data/MUTAG/Test/')
train_set = DataLoader(data_set[:100], batch_size, shuffle=True)
valid_set = DataLoader(data_set[100:300], batch_size, shuffle=False)

data_list = comp_model.train_model(train_set, valid_set, num_epoch, batch_size)

title = "learning rate: 1e-4, epochs: 500"
labels = ['Train Classification Loss', 'Train Accuracy', '', 'Test Accuracy']
dp.display_one(data_list, num_epoch, labels, title)
dp.display(data_list, num_epoch, labels, title)
