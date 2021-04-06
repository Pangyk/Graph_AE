# Graph Autoencoder for Graph Compression and Representation Learning
![MIAGAE model structure](https://github.com/Pangyk/Graph_AE/blob/main/Fig-1.png "MIAGAE model structure")

## About this repository
This repository contains an implementation of the models introduced in the paper 
[Graph Autoencoder for Graph Compression and Representation Learning](https://openreview.net/pdf?id=Bo2LZfaVHNi) 
by Ge et al. The model contains two modules. The compression module is an Autencoder which 
takes a graph as input and compresses the graph on the latent space. The classification module
takes the latent space produced by the compression module and predicts the class the graph belongs to.

The network is implemented using [PyTorch](https://pytorch.org/) and 
[PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/); and the rest of the framework is in Python.

## Dependencies
To get started with the framework, install the following dependencies:
- Python 3.7
- [PyTorch](https://pytorch.org/get-started/locally/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) depends on CUDA version
- [numpy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- CUDA 9.2, 10.1, 10.2, 11.0, or 11.1

## Dataset Preparation
1. You can use any dataset provided from [https://chrsmrrs.github.io/datasets/docs/datasets/](https://chrsmrrs.github.io/datasets/docs/datasets/).
Download the dataset and unzip it to the 'data' folder.
2. Or you can copy the name of the dataset, e.g. alchemy_full, and use the command line that will be introduced in section 'Quick Start' to download the dataset.
    ```
    python train_compression.py --d alchemy_full
    ```
3. Or if you have a customized dataset:  
    - Create a folder under 'data' folder. The name of folder should be your dataset's name.  
    - Create a subfolder called 'raw' under the folder you created.  
    - Produce the following files in the raw:  
        - {dataset name}_A.txt, a file indicating the adjacency matrix, in the format of v1, v2
        - {dataset name}_node_attributes.txt, a file indicating the features of the node.
        - {dataset name}_graph_labels.txt (optional), a file indicating the class of a graph.
        - You can find samples of these files by downloading any of the dataset from [https://chrsmrrs.github.io/datasets/docs/datasets/](https://chrsmrrs.github.io/datasets/docs/datasets/).
4. You can visit [http://ilab.usc.edu/datasets/i2sg](http://ilab.usc.edu/datasets/i2sg), which introduces how to generate graphs from images.

## Quick Start
1. Make sure you have installed all the dependencies.

2. Run the following command to train and save the compression module.

   ```
   python train_compression.py
   There are several options:
   
   # General options
    --m, name of model, default MIAGAE, you can choose from MIAGAE, UNet, Gpool, and SAGpool
    --device, default cuda, you can choose from cuda / cpu / cuda:0, etc.
   # Training specific options:
    --d, name of dataset, default FRANKENSTEIN
    --batch, batch size, default 512
    --e, number of epochs, default 100
    --lr, learning rate, default 1e-3
    --model_dir, path to save model, default /data/model/
    --n_train, number of samples for train set, default 3000
    --n_test, number of samples for test set, default 1000
   # Model specfic options:
    --k, number of kernels for each layer, default 2
    --depth, depth of encoder and decoder, default 3
    --c_rate, compression ratio for each layer of encoder, default 0.8
    --shapes, shape of each layer in encoder (decoder automatically inverts), default 64,64,64
   ```

3. Run the following command to train and test the classification module. (You should train the compression module first)
    ```
   python train_classifier.py
   There are several options:
   
   # Parameters that should be the same with compression module:
    --m, name of model
    --device
    --model_dir, path to save model, default /data/model/
    --k, number of kernels for each layer
    --depth, depth of encoder and decoder
    --c_rate, compression ratio for each layer of encoder
    --shapes, shape of each layer in encoder (decoder automatically inverts)
   
   # Training specific options:
    --d, name of dataset, default FRANKENSTEIN
    --batch, batch size, default 1024
    --e, number of epochs, default 100
    --lr, learning rate, default 1e-3
    --n_skip, skip some number of samples, default 2000
    --n_train, number of samples for train set, default 3000
    --n_test, number of samples for test set, default 1000
   # Model specfic options:
    --hidden, shape of the first layer in encoder, default 256
    --dropout, dropout rate, default 0.1
   ```

## Folder Introduction
- train (discarded): contains files to train different models. The 'train1' folder is similar to this folder, also discarded.

- pipeline: contains codes that generate scene graphs from images.

- classification: contains different models.
    - Baseline: simple classifier without Autoencoder.
    - Classifier: simple classifier for latent space produced by Autoencoder-based models.
    - Gpool_model: Autoencoder using G-pooling.
    - Graph_AE: Autoencoder using MIAGAE.
    - Graph_AE_SAGE: Autoencoder using multi-kernel SAGE.
    - SAG_model: Autoencoder using SAG pool.
    - UNet: Graph UNet.

- graph_ae: contains extra implementation for the models.

- utils: contains tools for visualization and loading pre-trained models.

## Citing This Work
If you find this work useful in your research, please consider citing:  
    
