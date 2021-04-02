"""
@Author: Shana
@File: Global_Dict.py
@Time: 2/21/21 00:31 AM
"""

import numpy as np
import argparse
import os

"""
Usage:
1. Put All the label files in one folder, and all the dict files in another folder(can't be the same folder)
    Please Name the label and dict files from same dataset with the same number suffix, For example:
    Dataset1: labels43.npy    dict43.npy
    Dataset2: labels22.npy    dict22.npy
2. Run command:
    python Global_Dict.py --outdir $Out --dict $Dict --label $Label
    for example:
        python Global_Dict.py --out /home/shana/result --dict /home/shana/dict --label /home/shana/label
3. Output
    The final output will be one global_dict.npy file as the global dict and n xx.npy files with the same name as the original label files
    For example, with datasets in 1 as input, the final output will be:
    global_dict.npy, labels43.npy, labels22.npy

"""


def main(args):
    Label_P = args.label
    Dict_P = args.dict
    Out_P = args.out_dir

    Ori_Label_N = []
    New_Label = []

    Ori_Dict = []
    New_Dict = []
    # Step 1: Load files
    for file in sorted(os.listdir(Label_P)):
        print(file)
        Ori_Label_N.append(file.split(".")[0])
        New_Label.append(np.load(os.path.join(Label_P, file)).tolist())

    for file in sorted(os.listdir(Dict_P)):
        print(file)
        Ori_Dict.append(np.load(os.path.join(Dict_P, file)).tolist())

    # print(Ori_Dict)
    # print(Ori_Label_N)
    # print(New_Label)

    ## Step 2: Get New Dict
    for i in range(len(Ori_Dict)):
        for j in range(len(New_Label[i])):
            New_Label[i][j] = Ori_Dict[i][New_Label[i][j] - 1]  # Nodel label starts from 1
        for j in range(len(Ori_Dict[i])):
            if Ori_Dict[i][j] not in New_Dict:
                New_Dict.append(Ori_Dict[i][j])

    ## Step 3: Get New Label:
    for i in range(len(New_Label)):
        for j in range(len(New_Label[i])):
            for k in range(len(New_Dict)):
                word_now = New_Label[i][j]
                if word_now == New_Dict[k]:
                    New_Label[i][j] = k + 1  # Nodel Label Starts from 1

    ## Save:
    # print(New_Dict)
    # print(New_Label)
    New_Dict = np.save(os.path.join(Out_P, 'global_dict.npy'), np.array(New_Dict))

    for i in range(len(New_Label)):
        label_now = np.array(New_Label[i])
        np.save(os.path.join(Out_P, Ori_Label_N[i] + '.npy'), label_now)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Global_Dict generator')
    parser.add_argument('--out_dir', type=str, default='',
                        help="Output dir for result")
    parser.add_argument('--label', type=str, default='', help="path to label files")
    parser.add_argument('--dict', type=str, default='', help="path to dict files")
    args = parser.parse_args()
    main(args)
