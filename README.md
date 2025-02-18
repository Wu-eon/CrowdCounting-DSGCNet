# DSGC-Net: A Dual-Stream Graph Convolution-Enhanced Network for Crowd Counting via Feature Correlation Mining
## Overview 🔍
### Abstract - 
Deep learning-based crowd counting methods have achieved remarkable progress in recent years. To address these challenges, we propose DSGC-Net, a Dual-Stream Graph Convolution-Enhanced Network based on feature correlation mining. DSGC-Net introduces the Density Approximation (DA) branch and the Representation Approximation (RA) branch. By constructing and modeling semantic graphs, it captures the inconsistencies in density variations and representation distributions.The DA branch incorporates a density prediction module that generates density distribution maps and constructs a density-driven semantic graph based on density similarity. The RA branch establishes a representation-driven semantic graph by computing global representation similarity.Then, Graph Convolutional Networks are used to model the features of the two semantic graphs separately, mining latent semantic relationships, which enhances the model’s ability to adapt to density variations and improves counting accuracy in multi-view and multi-pose scenarios. Extensive experiments on three benchmarks demonstrate that DSGC-Net outperforms state-of-the-art methods. In particular, we achieved MAE of 48.9 and 5.9 in ShanghaiTech Part A and B datasets respectively.


## Datasets 📚
