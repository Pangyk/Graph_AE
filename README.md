## Dir Intro
### 1. data dir: store data loaded from website.

### 2. gae dir: implement edge compression.
  * Edge_Comp.py is used to compress edges and evaluate it.
  * tmp.py is cloned from a github implementation of gae.

### 3. classification dir: implement classification task.
  * Baseline.py: baseline model, 3 GCN convs and 1 linear layer.
  * Classifier.py: our model, classification part, 2 linear layers.
  * Comp_Cfy.py: our model, compression part, 3 GAT convs for encoder, 3 GAT convs for mask layer, 3 GAT layers for decoder.
  * Node_Comp.py: discarded.

### 4. model dir: implement compression models
  * Baseline.py: baseline model, with only MSE loss, no penalty, does not compress graphs.
  * Node_Comp_Mask.py: model using mask to select nodes. Penalty func = (abs(mean - desired keep rate) - var) if epoch > 100 else -var
  * Node_Comp_Norm.py: model using thresh hold to select nodes. Penalty func = min(sum(features) - thresh hold, 0)
  * Node_Comp_Pretrain.py: model using pretrain. Penalty func = min(sum(features) - thresh hold, 0)
  * Node_Comp_Upper.py: original design. Penalty func = -var
  * Node_Comp_Dist.py: discarded.
  
### 5. tmp dir: discarded

### 6. utils dir: tools for loading dataset, display plots and statistics in training process.
  * CustomDataSet.py: load dataset. current class we are using in classification task is 'SelectGraph'
  * Display_Plot.py: display plots. Usually MSE loss, penalty loss, total loss, and remaining nodes number.
  * Display_Statistic.py: display statistics. How many nodes remain in a graph, which graphs has remaining nodes, edge info, and feature info.

## Train
1. Train_D_xxx means train & display, just train a model and display its MSE loss, penalty loss, total loss, and remaining nodes number.
2. Train_C_xxx means train & classify, used to deal with classification tasks and display classification loss, train accuracy and test accuracy.
3. Train_D_Base means training baseline model, Train_D means training our model.
4. Train_C_Base means training baseline model, Train_C is discarded.
5. Separate_Train is first pretrain a compression model and then train a classification model on a dataset with different labels.

## Dataset
1. Current dataset is COLORS-3. The data is from Synthetic. It contains 10500 graphs, with an average of 61.31 nodes and 91.03 edges in each graph.
2. The feature size is 5, and the feature is a combination of 0s and 1s, e.g. [1, 0, 1, 1, 0]. 
3. There are 11 classes. In Separate_Train.py, graphs labeled 0-7 are used to pretrain a compression model. Graphs labeled 8-10 are used to train classification model 
(for both Separate_Train.py and Train_C_Base.py)
