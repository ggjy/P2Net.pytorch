# -*- coding: utf-8 -*-
# 
# Human part module
# Edited by Jianyuan Guo 
# jyguo@pku.edu.cn
# 2019.10

import torch
import os
import sys
import pdb
from torch import nn
from torch.nn import functional as F
import functools
from torch.nn import init
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import random
import numpy as np
import matplotlib.cm as cm
from scipy.misc import imresize


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.constant(m.weight.data, 0.0)
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.constant(m.weight.data, 0.0)
        init.constant(m.bias.data, 0.0)


class MS_PartGuidedBlock(nn.Module):
    '''
    Compute the affinity based feature maps and segmentation map of two scales.
    Thus to get the context information of the specified scale.
    Input:
        N X C X H X W
    Parameters:
        in_channels    : the dimension of the input feature map
        value_channels : the dimension of W_g
        bn_layer       : whether use BN within W_rho
        scale          : choose the scale to downsample the input feature maps
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''
    def __init__(self, in_channels, value_channels, out_channels=None, mode='embedded_gaussian',
                 bn_layer=True, scale=1, vis=False, choice=1):
        super(MS_PartGuidedBlock, self).__init__()

        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation']
        self.vis = vis
        self.choice = choice
        self.mode = mode
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels

        self.pool = nn.AvgPool2d(kernel_size=(scale, scale))

    def forward(self, x, label, path=None):
        batch_size, h0, w0 = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)
        
        h,w = h0,w0
        if self.choice == 1:
            x = F.interpolate(input=x, size=((h,w)), mode='bilinear',align_corners=True)
            value = x.view(batch_size, self.in_channels, -1)
        elif self.choice == 2:
            x = self.value_2(x)
            x = F.interpolate(input=x, size=((h,w)), mode='bilinear',align_corners=True)
            value = x.view(batch_size, self.value_channels, -1)
        elif self.choice == 3:
            x = self.value_3(x)
            x = F.interpolate(input=x, size=((h,w)), mode='bilinear',align_corners=True)
            value = x.view(batch_size, self.value_channels, -1)
        
        value = value.permute(0, 2, 1)
        label = F.interpolate(input=label.unsqueeze(1).type(torch.cuda.FloatTensor), size=((h,w)), mode='nearest')

        label_row_vec = label.view(batch_size, 1, -1).expand(batch_size, h*w, h*w)
        label_col_vec = label_row_vec.permute(0, 2, 1)
        pair_label = label_col_vec.eq(label_row_vec)
        
        # background use global, commented by Huang Lang
        label_col_vec = 1-label_col_vec
        label_col_vec[label_col_vec<0]=0
        pair_label = pair_label.long() + label_col_vec.long()
        pair_label[pair_label>0]=1
        
        sim_map = F.normalize(pair_label.type(torch.cuda.FloatTensor), p=1, dim=2)
        
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        if self.choice == 1:
            context = context.view(batch_size, self.in_channels, *x.size()[2:])
            context = F.interpolate(input=context, size=((h0,w0)), mode='bilinear',align_corners=True)
        elif self.choice == 2:
            context = context.view(batch_size, self.value_channels, *x.size()[2:])
            context = F.interpolate(input=context, size=((h0,w0)), mode='bilinear',align_corners=True)
            context = self.W(context)
        elif self.choice == 3:
            context = context.view(batch_size, self.value_channels, *x.size()[2:])
            context = F.interpolate(input=context, size=((h0,w0)), mode='bilinear',align_corners=True)
            context = self.W(context)

        return context


class MSPartGuidedModule(nn.Module):
    """
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: specify the dropout ratio
        fusion: We provide two different fusion method, "concat" or "add"
        sizes: compute the attention based on diverse scales based context
    Return:
        features after "concat" or "add"
    """
    def __init__(self, in_channels, out_channels, value_channels, fusion="concat", sizes=(1,4,8,16), vis=False, choice=1):
        super(MSPartGuidedModule, self).__init__()

        self.fusion = fusion
        self.stages = []
        self.group = len(sizes)
        self.stages = nn.ModuleList([MS_PartGuidedBlock(in_channels, value_channels, in_channels//self.group, scale=size, vis=vis, choice=choice) for size in sizes])
        self.bottleneck_add = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.LeakyReLU(0.5)
            )
        self.bottleneck_concat = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels),
            #nn.ReLU()
            )
        self.bottleneck_concat.apply(weights_init_kaiming)
        self.bottleneck_add.apply(weights_init_kaiming)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, feats, label, path=None):
        priors = [stage(feats, label, path) for stage in self.stages]

        if self.fusion == "concat":
            context = feats
            for i in range(len(priors)):
                context = torch.cat([context, priors[i]], 1)
            bottle = self.bottleneck_concat(context)
        elif self.fusion == 'add':
            context = priors[0]
            for i in range(1, len(priors)):
                context += priors[i]
            bottle = self.bottleneck_add(context + feats)
        elif self.fusion == '+':
            context = [priors[0]]
            for i in range(1, len(priors)):
                context += [priors[i]]
            bottle = self.bottleneck_add(torch.cat(context, 1)) + feats

        return bottle


if __name__=='__main__':
    pass