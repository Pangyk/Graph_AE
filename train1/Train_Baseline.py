import argparse

from torch_geometric.data import DataLoader
from utils.CustomDataSet import SceneGraphs, SelectGraph
from classification.Baseline import Net
import utils.Display_Plot as dp
import torch
import os


def main(arg):
    batch_size = arg.batch
    num_epoch = arg.e
    device = torch.device(arg.device)

    model = Net.get_instance().to(device)

    SelectGraph.data_name = arg.d
    data_set_Shana = SelectGraph(os.path.join('data', SelectGraph.data_name))
    train_set = DataLoader(data_set_Shana[2000:2500], 500, shuffle=False)
    test_set = DataLoader(data_set_Shana[2500:3000], 500, shuffle=False)

    data_list = model.train_model(train_set, test_set, num_epoch, batch_size)

    title = "learning rate: " + arg.lr + ", epochs: " + num_epoch
    labels = ['MSE Loss', 'Penalty Loss', 'Total Loss', 'Test Acc']
    dp.display_one(data_list, num_epoch, labels, title)
    dp.display(data_list, num_epoch, labels, title)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Global_Dict generator')
    parser.add_argument('--d', type=str, default='Shana',
                        help="dataset name")
    parser.add_argument('--m', type=str, default='', help="path to label files")
    parser.add_argument('--device', type=str, default='cuda', help="cuda / cpu")
    parser.add_argument('--batch', type=int, default=512, help="batch size")
    parser.add_argument('--e', type=int, default=100, help="number of epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    args = parser.parse_args()
    main(args)
