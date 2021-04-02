from torch_geometric.data import DataLoader
from utils.CustomDataSet import SelectGraph, SceneGraphs
from classification.UNet import Net
from classification.Classifier import MLP
import utils.Display_Plot as dp
import utils.load_model as lm
import torch

device = torch.device('cuda')

num_epoch = 100
batch_size = 500
comp_model = Net.get_instance().to(device)
cfy_model = MLP.get_instance().to(device)


SelectGraph.data_name = 'Shana7000'
data_set_Shana = SelectGraph('data/' + SelectGraph.data_name)
train_set2 = DataLoader(data_set_Shana[5000:6000], 1000, shuffle=False)
test_set = DataLoader(data_set_Shana[6000:7000], 1000, shuffle=False)
comp_model.load_state_dict(torch.load("../data/model/U/UNET_TRANSFER.ckpt"), strict=True)
group1, group2 = lm.load_model_result(comp_model, train_set2, test_set, device)
data_list2 = cfy_model.train_model(train_set2, test_set, int(num_epoch), group1, group2)

title = "GAE Normal MK2"
labels = ['Train Loss', 'Train Acc', title, 'Test Acc']
dp.display(data_list2, int(num_epoch), labels, title)
