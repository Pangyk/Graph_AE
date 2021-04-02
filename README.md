# Introduction
There are several folders in this project and the function of each folder is described below:
### 1. train
This folder contains files to train different models. The 'train1' folder is similar to this folder.

### 2. pipeline
This folder contains codes that generate scene graphs from images.

### 3. classification
This folder contains different models.
Baseline: simple classifier without Autoencoder.
Classifier: simple classifier for latent space produced by Autoencoder-based models.
Classifier_UNet: classifier for Graph UNet.
Gpool_model: Autoencoder using G-pooling.
Graph_AE: Autoencoder using MIAGAE.
Graph_AE_SAGE: Autoencoder using multi-kernel SAGE.
SAG_model: Autoencoder using SAG pool.
UNet: Graph UNet.

### 4. graph_ae
This folder contains extra implementation for the models.

### 5. utils
This folder contains tools for visualization and loading pre-trained models.
