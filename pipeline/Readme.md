#### Data_P.py

##### 1. parameters:

out_dir: path to output
name: the name you want to give to your dataset
root: path of custom_data_info.json,'custom_prediction.json','scene_validation_annotations_20170908.json'



##### 2. run example:

python Data_P.py --out_dir /home/shana/result --root /home/shana/test --name Shana

##### 3. output:

[file1]name_A.txt
[file2]name_graph_indicator.txt
[file3]name_graph_labels.txt
[file4]nodel_labels.npy
[file5]label_dict.npy


###### g_word_vec function

file4 stores each node's label
and file5 stores the corresponding word of each node
just finish Line30 to get the word embedding of each word

then save the node attributes in to [file6] name_node_attributes.txt

Format of [file6]:
2 nodes with feature[1,2,31] and [4,5,6]
it will be like
 1.0000000, 2.0000000, 31.000000
 4.0000000, 5.0000000, 6.0000000

Note there is a blank line at last.

#### How to use custom dataset in pytorch

1. run these code:

   SelectGraph.data_name = Name
   SelectGraph.thresh = 8
   SelectGraph.direction = -1
   data_set = SelectGraph(os.path.join('Data', SelectGraph.data_name))
   You will get an error said you can't download the dataset from internet

2. Move file1236 into Graph_AE/untils/data/name/name/raw
   For example, for dataset named shana,
   you should move all the files into Graph_AE/untils/data/shana/shana/raw

3. Comment the code that reports an error
   it's in /home/shana/anaconda3/envs/momo/lib/python3.6/site-packages/torch_geometric/datasets/tu_dataset.py on my computer.

Just comment the download function in tu_dataset.py
you could found the location from error message.


Then run the code in step 1 again. It should work (at least works on my computer :)


#### Complex.py

It's my code to test compression between GAE& U_net, and test classification between GAE & baseline

#### Class_Base.py

classification baseline which stores at Graph_AE/classification

#### U_net.py

modified to calculate saved edge weights.
located in Graph_AE/graph_ae

#### GraphUnet.py

modified to calculate saved edge weights.