# -*- coding: utf-8 -*-
# 
# Human part / Latent part / self-attention
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
import random
import numpy as np
import matplotlib.cm as cm


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.constant_(m.weight.data, 0.0)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 0.0)
        init.constant_(m.bias.data, 0.0)


class _MS_NonLocalBlockND(nn.Module):
    '''
    Modified based on the _NonLocalBlockND, compute the affinity based feature maps of two scales.
    thus to get the context information of the specified scale.
    Input:
        N X C X H X W
    Parameters:
        in_channels: the dimension of the input feature map
        c1         : the dimension of W_theta and W_phi
        c2         : the dimension of W_g and W_rho
        bn_layer   : whether use BN within W_rho
        use_g / use_w: whether use the W_g transform and W_rho transform
        scale      : choose the scale to downsample the input feature maps
    Return:
        N X C X H X W
    '''
    def __init__(self, in_channels, c1, c2, out_channels=None, mode='embedded_gaussian',
                 sub_sample=False, bn_layer=False, use_g=True, use_w=True, scale=1, vis=False):
        super(_MS_NonLocalBlockND, self).__init__()

        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation']
        self.vis = vis
        self.mode = mode
        self.use_g = use_g
        self.use_w = use_w
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = c1
        self.context_channels = in_channels
        if use_g:
            self.context_channels = c2
        if out_channels == None:
            self.out_channels = in_channels

        self.pool = nn.AvgPool2d(kernel_size=(scale, scale))
        self.theta = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0, bias=True)
            )
        self.phi = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                    kernel_size=1, stride=1, padding=0, bias=True)
            )
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.context_channels,
                kernel_size=1, stride=1, padding=0, bias=True)
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.context_channels, out_channels=self.out_channels,
                kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.out_channels)
        )
        self.theta.apply(weights_init_kaiming)
        self.phi.apply(weights_init_kaiming)
        self.g.apply(weights_init_kaiming)
        self.W.apply(weights_init_kaiming)

    def forward(self, x, path=None):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        x_scale = self.pool(x)
        if self.use_g:
            g_x = self.g(x_scale).view(batch_size, self.context_channels, -1)
        else:
            g_x = x_scale.view(batch_size, self.context_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x_scale).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x_scale).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f = (self.inter_channels**-.5) * f
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.context_channels, *x.size()[2:])

        if self.use_w:
            W_y = self.W(y)
        else:
            W_y = y
        return W_y


class MS_NONLocalBlock2D(_MS_NonLocalBlockND):
    def __init__(self, in_channels, c1=None, c2=None, out_channels=None, mode='embedded_gaussian', 
        bn_layer=False, use_g=True, use_w=True, scale=1, vis=False):
        super(MS_NONLocalBlock2D, self).__init__(in_channels,
                                              c1=c1,
                                              c2=c2,
                                              out_channels=out_channels,     
                                              mode=mode,
                                              bn_layer=bn_layer,
                                              use_g=use_g,
                                              use_w=use_w,
                                              scale=scale,
                                              vis=vis)


class MSPyramidAttentionContextModule(nn.Module):
    """
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: specify the dropout ratio
        fusion: We provide two different fusion method, "concat" or "add"
        sizes: compute the attention based on diverse scales based context
    Return:
        features after "concat" or "add"
    """
    def __init__(self, in_channels, out_channels, c1, c2, dropout=0, fusion="concat", sizes=(1,4,8,16), use_head_bn=False, if_gc=0, vis=False, norm=0):
        super(MSPyramidAttentionContextModule, self).__init__()
        self.norm = norm
        self.if_gc = if_gc
        self.fusion = fusion
        self.stages = []
        self.group = len(sizes)
        self.c1 = c1
        self.c2 = c2 
        self.stages = nn.ModuleList([self._make_stage(in_channels, self.c1, self.c2, in_channels//self.group, size, use_head_bn=use_head_bn, vis=vis) for size in sizes])
        self.bottleneck_add = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout)
            )
        if self.if_gc == 1:
            channel_in = 3
        else:
            channel_in = 2
        self.bottleneck_concat = nn.Sequential(
            nn.Conv2d(in_channels*channel_in, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout)
            )
        self.bottleneck_add.apply(weights_init_kaiming)
        self.bottleneck_concat.apply(weights_init_kaiming)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.W = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                    kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(in_channels)
            )
        self.W.apply(weights_init_kaiming)

    def _make_stage(self, in_channels, c1, c2, out_channels, size, use_head_bn, vis):
        return MS_NONLocalBlock2D(in_channels=in_channels, c1=c1, c2=c2, out_channels=out_channels, mode='dot_product', use_g=True, use_w=True, scale=size, bn_layer=use_head_bn, vis=vis)
    
    def forward(self, feats, parsing=None, path=None):
        if parsing is not None:
            batch_size, channel, h, w = feats.size(0), feats.size(1), feats.size(2), feats.size(3)
            x = F.interpolate(input=feats, size=((h,w)), mode='bilinear',align_corners=True)
            value = x.view(batch_size, channel, -1)
            value = value.permute(0, 2, 1)
            label = F.interpolate(input=parsing.unsqueeze(1).type(torch.cuda.FloatTensor), size=((h,w)), mode='nearest')
            label_row_vec = label.view(batch_size, 1, -1).expand(batch_size, h*w, h*w)
            label_col_vec = label_row_vec.permute(0, 2, 1)
            pair_label = label_col_vec.eq(label_row_vec)
            sim_map = F.normalize(pair_label.type(torch.cuda.FloatTensor), p=1, dim=2)
            context = torch.matmul(sim_map, value)
            context = context.permute(0, 2, 1).contiguous()
            context = context.view(batch_size, channel, *x.size()[2:])
            context = F.interpolate(input=context, size=((h,w)), mode='bilinear',align_corners=True)
            parsing = self.W(context)

        if self.norm > 0:
            feats_norm =  self.norm * F.normalize(feats, p=2, dim=1) 
            priors = [stage(feats_norm) for stage in self.stages]
        else:
            priors = [stage(feats, path) for stage in self.stages]

        if self.if_gc >= 1:
            batch_size, c, h, w = feats.size(0), feats.size(1), feats.size(2), feats.size(3)
            gc = self.avgpool(feats)
            if self.norm > 0:
                gc = self.norm * F.normalize(gc, p=2, dim=1)
            gc = gc.repeat(1,1,h,w)

        if self.fusion == "concat":
            context = feats
            for i in range(len(priors)):
                context = torch.cat([context, priors[i]], 1)
            if self.if_gc == 1:
                bottle = self.bottleneck_concat(torch.cat([context, 0.5*gc], 1))
            else:
                bottle = self.bottleneck_concat(context) # torch.cat([context, parsing], 1)
            return bottle
        elif self.fusion == 'add':
            context = priors[0]
            for i in range(1, len(priors)):
                context += priors[i]
            bottle = self.bottleneck_add(context + feats)
        elif self.fusion == '+':
            context = [priors[0]]
            for i in range(1, len(priors)):
                context += [priors[i]]
            if self.if_gc == 1:
                if parsing is not None:
                    bottle = torch.cat(context, 1) + parsing + feats + gc
                else:
                    bottle = torch.cat(context, 1) + feats + gc
            else:
                if parsing is not None:
                    bottle = torch.cat(context, 1) + parsing + feats
                else:
                    bottle = torch.cat(context, 1) + feats
                    
        return bottle