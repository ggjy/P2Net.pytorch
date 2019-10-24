# P2Net
  This repository is the codes for ICCV2019 paper [Beyond Human Parts: Dual Part-Aligned Representations for Person ReID](https://arxiv.org/pdf/1910.10111.pdf)
  
I will further update the code after cvpr deadline :). 

This repo can reproduce our results on DukeMTMC-reID.

## Prerequisites

- Python 3.6
- GPU Memory >= 6G
- Numpy
- Pytorch >= 0.4
- Torchvision >= 0.2.0

## DukeMTMC-reID

### Dataset & Preparation
Download DukeMTMC-ReID Dataset.

Preparation: You may need our generated human part masks [BaiDuYun](https://pan.baidu.com/s/18IIrRSnRN97mC8IShlmXwQ).
Remember to change the dataset path to your own path in duke.py.

### Train
Train a model by
```bash
cd scripts
sh resnet50_softmax.sh
```

### Results

The model is based on Resnet50. Input images are resized to 384x128.

**Note that the result may be better than Table 9 in the paper. (Setting here is batchsize 48 on 1 GPU)**

| Method | Rank-1 | Rank-5 | Rank-10 | mAP | Model |
| --------- | ----- | ----- | ----- | ----- | ----- |
| Baseline | 81.10 | 89.59 | 92.19 | 64.87 |[BaiDuYun](https://pan.baidu.com/s/1JZ_fHiqXjNDtWearwEIQ3Q) |
|1 x Latent | 82.92 | 91.03 | 93.49 | 67.09 |[BaiDuYun](https://pan.baidu.com/s/1rvPB_-hOB8huqWTJuBDYSw) |
|1 x DPB | 84.83 | 92.28 | 94.08 | 68.62 |[BaiDuYun](https://pan.baidu.com/s/1BSb51t8iIihyzKAyLcOgLQ) |

## Citation
```
@InProceedings{Guo_2019_ICCV,
author = {Guo, Jianyuan and Yuan, Yuhui and Huang, Lang and Zhang, Chao and Yao, Jin-Ge and Han, Kai},
title = {Beyond Human Parts: Dual Part-Aligned Representations for Person Re-Identification},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
```

## Acknowledgement
