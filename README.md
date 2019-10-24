# P2Net
  This repository is the codes for ICCV2019 paper [Beyond Human Parts: Dual Part-Aligned Representations for Person ReID](https://arxiv.org/pdf/1910.10111.pdf)

## DukeMTMC-reID
The model is based on Resnet50. Input images are resized to 384x128.

**Note that the result may be better than Table 9 in the paper. (Setting here is batchsize 48 on 1 GPU)**

| Method | Rank-1 | Rank-5 | Rank-10 | mAP | Model |
| --------- | ----- | ----- | ----- | ----- | ----- |
| Baseline | 81.10 | 89.59 | 92.19 | 64.87 |
|1 \times Latent | 82.92 | 91.03 | 93.49 | 65.09 |
|1 \times DPB | 84.83 | 92.28 | 94.08 | 68.62 |

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
