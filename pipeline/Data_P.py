"""
@Author: Shana
@File: Data_P.py
@Time: 2/18/21 12:15 PM
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def g_word_vec(root, out, Name):
    """
    :param root: path to nodel_labels.npy & label_dict.npy
    :param out: path to save the result
    :param Name: dataset name
    :return:
    """
    O_Natrr = Name + '_node_attributes.txt'
    label_dict = np.load(os.path.join(root, 'global_dict.npy'))
    node_labels = np.load(os.path.join(root, 'node_labels.npy'))

    #  get word vector for each word in the label_dict
    result = dict()
    max_v = len(label_dict) + 1
    setup_seed(max_v)
    print(max_v)
    embeds = nn.Embedding(max_v, 500)
    np.set_printoptions(formatter={'float': '{: 0.6f}'.format})
    for idx in range(max_v):
        embed_idx = Variable(torch.LongTensor([idx]))
        result[idx] = embeds(embed_idx)[0].detach().cpu().numpy()

    # get the corresponding node feature
    with open(os.path.join(out, O_Natrr), 'w+') as f:
        for i in node_labels:
            s = str(list(result[int(i)]))
            s = " " + s[1: -1]
            f.write(s + "\n")
    f.close()


def main(args):
    Info = "custom_data_info.json"
    Pred = 'custom_prediction.json'
    Ann = 'scene_validation_annotations_20170908.json'
    Root = args.root
    Out = args.out_dir
    Name = args.name
    print(Root)
    print(Out)
    print(Name)
    O_Edge = Name + '_A.txt'
    O_Gid = Name + '_graph_indicator.txt'
    O_Gla = Name + '_graph_labels.txt'

    # Num_Nodes = 79
    # Num_Edges = int(79 * 79 * 0.1)
    Num_Edges = int(79 * 79 * 0.1)
    Edges = []
    Node_Labels = []
    Node_Index = []
    Graph_Labels = []

    a = open(os.path.join(Root, Info), 'r')
    info = json.load(a)
    print("a")
    a = open(os.path.join(Root, Pred), 'r')
    pred = json.load(a)
    print("b")
    a = open(os.path.join(Root, Ann), 'r')
    ann = json.load(a)
    print("c")

    Num = len(info['idx_to_files'])
    Node_L = info['ind_to_classes']

    global_node_index = 1
    for i in range(Num):
        print("======================")
        ## Step 1: Get Graph Label
        filename = info['idx_to_files'][i].split('/')
        filename = filename[len(filename) - 1]
        for j in range(len(ann)):
            if ann[j]['image_id'] == filename:
                Graph_Labels.append(ann[j]['label_id'])

        pr_now = pred[str(i)]
        node_temp = []
        Edges_temp = []
        ## Step 2: Get_temp_Edges
        pairs = pr_now['rel_pairs']
        Num_Edges = min(Num_Edges, len(pairs))
        for j in range(Num_Edges):
            a = pairs[j][0]
            b = pairs[j][1]
            Edges_temp.append([a, b])
            # print(str(a)+" "+str(b))
            if not a in node_temp:
                node_temp.append(a)
            if not b in node_temp:
                node_temp.append(b)

        Num_Nodes = len(node_temp)
        # print(Num_Nodes)
        # print(node_temp)
        ## Step 3: Get Node and real edges
        for j in range(Num_Nodes):
            # node_index.append(j)
            Node_Labels.append(pr_now['bbox_labels'][node_temp[j]])
        for j in range(len(Edges_temp)):
            tp1 = Edges_temp[j][0]
            tp2 = Edges_temp[j][1]
            # print("shh"+str(tp1) + " " + str(tp2))
            for k in range(Num_Nodes):
                if tp1 == node_temp[k]:
                    a = k
                if tp2 == node_temp[k]:
                    b = k
            # print(str(a+global_node_index) + " " + str(b+global_node_index))
            Edges.append([a + global_node_index, b + global_node_index])

        # Node_Labels.extend(pr_now['bbox_labels'][:Num_Nodes])

        ## Step 4: Indicate Node
        Num_Nodes = len(node_temp)
        for j in range(Num_Nodes):
            Node_Index.append(i + 1)
        global_node_index += Num_Nodes
        print(global_node_index)

    # Save
    with open(os.path.join(Out, O_Edge), 'w+') as f:
        for i in Edges:
            f.write(str(i[0]) + ", " + str(i[1]) + "\n")
    f.close()

    with open(os.path.join(Out, O_Gid), 'w+') as f:
        for i in Node_Index:
            f.write(str(i) + "\n")
    f.close()

    with open(os.path.join(Out, O_Gla), 'w+') as f:
        for i in Graph_Labels:
            f.write(str(i) + "\n")
    f.close()

    a = np.array(Node_Labels)
    b = np.array(Node_L)
    np.save(os.path.join(Out, 'node_labels.npy'), a)
    np.save(os.path.join(Out, 'label_dict.npy'), b)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data processing for Scene Graph')
    parser.add_argument('--out_dir', type=str, default='',
                        help="Output dir for result")
    parser.add_argument('--root', type=str, default='', help="Scene Result files location")
    parser.add_argument('--name', type=str, default='shana', help="Name of our dataset")
    args = parser.parse_args()
    main(args)
