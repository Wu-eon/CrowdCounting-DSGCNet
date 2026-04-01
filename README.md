<h1 align="center">DSGC-Net: A Dual-Stream Graph Convolutional Network for Crowd Counting via Feature Correlation Mining (PRCV 2025)</h2>

<p align="center">
  <a href="https://arxiv.org/abs/2509.02261">
    <img src="https://img.shields.io/badge/arXiv-2509.02261-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://link.springer.com/chapter/10.1007/978-981-95-5758-5_29">
    <img src="https://img.shields.io/badge/PRCV-2025-blue.svg" alt="PRCV 2025">
  </a>
</p>

## Overview🔍
<div>
    <img src="assets/framework.jpg" width="95%" height="95%">
</div>


**Figure 1. The overall framework of the proposed Dual-Stream Graph Convolutional Network (DSGC-Net).**

**Abstract -** Deep learning-based crowd counting methods have achieved remarkable progress in recent years. However, in complex crowd scenarios, existing models still face challenges when adapting to significant density distribution differences between regions. Additionally, the inconsistency of individual representations caused by viewpoint changes and body posture differences further limits the counting accuracy of the models. To address these challenges, we propose DSGC-Net, a Dual-Stream Graph Convolutional Network based on feature correlation mining. DSGC-Net introduces a Density Approximation (DA) branch and a Representation Approximation (RA) branch. By modeling two semantic graphs, it captures the potential feature correlations in density variations and representation distributions. The DA branch incorporates a density prediction module that generates the density distribution map, and constructs a density-driven semantic graph based on density similarity. The RA branch establishes a representation-driven semantic graph by computing global representation similarity. Then, graph convolutional networks are applied to the two semantic graphs separately to model the latent semantic relationships, which enhance the model's ability to adapt to density variations and improve counting accuracy in multi-view and multi-pose scenarios. Extensive experiments on three widely used datasets demonstrate that DSGC-Net outperforms current state-of-the-art methods. In particular, we achieve MAE of 48.9 and 5.9 in ShanghaiTech Part A and Part B datasets, respectively.

<div>
    <img src="assets/vis.jpg" width="95%" height="95%">
</div>

**Figure 2. Visualization examples. 1) The ground truth. 2) Prediction results of the proposed DSGC-Net. 3) The outputs of the Density Prediction (DP) module. 4) and 5) are the detailed comparisons of the prediction results of the baseline and the proposed DSGC-Net.**

## Datasets📚
We evaluate the proposed method on three of the most widely used crowd counting datasets: ShanghaiTech Part A and Part B and UCF-QNRF.
## Experimental Results🏆
### Comparison with State-of-the-Art Methods

| Method      | Venue       | Backbone     | SHTech Part A (MAE/MSE) | SHTech PartB (MAE/MSE) | UCF-QNRF (MAE/MSE) |
|-------------|-------------|--------------|------------------------|------------------------|--------------------|
| CSRNet      | CVPR-2018   | VGG-16       | 68.2 / 115.0           | 10.6 / 16.0            | - / -              |
| BL          | ICCV-2019   | VGG-19       | 62.8 / 101.8           | 7.7 / 12.7             | 88.7 / 154.8       |
| DM-Count    | NeurIPS-2020| VGG-19       | 59.7 / 95.7            | 7.4 / 11.8             | 85.6 / 148.3       |
| HYGNN       | AAAI-20     | VGG-16       | 60.2 / 94.5            | 7.5 / 12.7             | 100.8 / 185.3      |
| P2PNet      | ICCV-2021   | VGG-16       | 52.7 / 85.1            | 6.2 / 9.9              | 85.3 / 154.5       |
| TopoCount   | AAAI-2021   | VGG-16       | 61.2 / 104.6           | 7.8 / 13.7             | 89.0 / 159.0       |
| LSC-CNN     | TPAMI-2021  | VGG-16       | 66.4 / 117.0           | 8.1 / 12.7             | 120.5 / 218.2      |
| CLTR        | ECCV-2022   | DETR         | 56.9 / 95.2            | 6.5 / 10.6             | 85.8 / 141.3       |
| Ctrans-MISN | PRAI-2022   | ViT          | 55.8 / 95.9            | 7.3 / 11.4             | 95.2 / 180.1       |
| NDConv      | SPL-2022    | ResNet-50    | 61.4 / 104.2           | 7.8 / 13.8             | 91.2 / 165.6       |
| AutoScale   | IJCV-2022   | VGG-16       | 65.8 / 112.1           | 8.6 / 13.9             | 104.4 / 174.2      |
| PTCNet      | EAAI-2023   | PVT          | <u>51.7</u> / 79.6      | 6.3 / 10.6             | <u>79.7</u> / **133.2** |
| GMS         | TIP-2023    | HRNet        | 68.8 / 138.6           | 16.0 / 33.5            | 104.0 / 197.4      |
| DMCNet      | WACV-2023   | VGG-16       | 58.5 / 84.6            | 8.6 / 13.7             | 96.5 / 164.0       |
| VMambaCC    | arXiv-2024  | Mamba        | 51.9 / 81.3            | 7.5 / 12.5             | 88.4 / 144.7       |
| DDRANet     | SPL-2024    | VGG-16       | 52.1 / <u>78.4</u>      | 6.9 / 10.3             | 89.2 / 146.9       |
| CAAPN       | TPAMI-2024  | VGG-16       | 54.4 / 97.3            | **5.8** / <u>9.8</u>    | 83.9 / 144.3       |
|CrowdFPN     | Ap.Int.-2025 | Twins     | 52.5 / 88.5            | 6.5 / 9.9               | 81.2 / 157.3       |
| **Ours**    | -           | VGG-16       | **48.9** / **77.8**    | <u>5.9</u> / **9.3**    | **79.3** / <u>133.9</u> |

**Bold** indicates the best performance.


## Getting Started🚀
### 1. Data Preparation
First, prepare the dataset. Then, run the following command to generate the corresponding ground truth density distribution maps:
```bash
python density_data_preparation/k_nearest_gaussian_kernel.py
```
Immediately afterward, We use a list file to collect all the input images and their ground truth annotations in a counting dataset. When your dataset is organized as recommended in the following, the format of this list file is defined as:
```
train/scene01/img01.jpg train/scene01/img01.txt
train/scene01/img02.jpg train/scene01/img02.txt
...
train/scene02/img01.jpg train/scene02/img01.txt
```
#### Dataset structures:
```
DATA_ROOT/
        |->train/
        |    |->scene01/
        |    |->scene02/
        |    |->...
        |->test/
        |    |->scene01/
        |    |->scene02/
        |    |->...
        |->train.list
        |->test.list
        |->gt_density_maps/
        |    |->train/
        |    |->test/  
```
### 2. Training
The network can be trained using the train.py script. For training on ShanghaiTech PartA, use
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --data_root $DATA_ROOT \
    --dataset_file SHHA \
    --epochs 3500 \
    --lr_drop 3500 \
    --output_dir ./logs \
    --checkpoints_dir ./weights \
    --tensorboard_dir ./runs \
    --lr 0.0001 \
    --lr_backbone 0.00001 \
    --batch_size 8 \
    --eval_freq 1 \
    --gpu_id 0
```
### 3. Test
You can prepare the data according to the previously mentioned dataset structures and directly use our trained weights to predict individuals' locations and the total crowd count (which will be displayed on the predicted images). Please run
```bash
python run_test.py
```
## Friendly reminder😊
The repository is gradually being improved. If you need further assistance, please contact us. Feedback and suggestions are also welcome.😀
## Cite our work📝
```Coming soon...```
## License📜
The source code is free for research and education use only. Any comercial use should get formal permission first.
