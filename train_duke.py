# -*- coding: utf-8 -*-
from __future__ import print_function, division

import argparse
import json
import os
import pdb
import sys
import scipy.io
import time
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
from torch.autograd import Variable
from torchvision import datasets, models, transforms

from utils import RandomIdentitySampler, logging, RandomErasing
from utils.test_utils import *
from model import P2Net
from duke import *
from blocks import TripletLoss


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1')
parser.add_argument('--name',default='ResNet50', type=str, help='output model name')
parser.add_argument('--batchsize', default=48, type=int, help='batchsize')
parser.add_argument('--block_num', default=0, type=int, help='dual part block number')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='training weight decay')
parser.add_argument('--PCB', action='store_true', help='if use PCB+ResNet50' )
parser.add_argument('--era', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--fusion', default='concat', type=str, help='self-attention-choice')
parser.add_argument('--loss', default='softmax', type=str, help='choice of loss')
parser.add_argument('--layer', default='50', type=str, help='Resnet-layer')
parser.add_argument('--num_instances', default=8, type=int, help='for triplet loss')
parser.add_argument('--epoch', default=60, type=int, help='training epoch')
parser.add_argument('--margin', default=4, type=float, help='triplet loss margin')
parser.add_argument('--file_name', default='result', type=str, help='file name to save')
parser.add_argument('--pmap', default=False, help='use part_map')
parser.add_argument('--mat', default='', type=str, help='name for saving representation' )
opt = parser.parse_args()

sys.stdout = logging.Logger(os.path.join('/home/guojianyuan/ReID_Duke/'+opt.file_name+'/'+opt.name+'/', 'log.txt'))
tripletloss = TripletLoss(opt.margin)

gpu_ids = []
str_gpu_ids = opt.gpu_ids.split(',')
for str_id in str_gpu_ids:
    gpu_ids.append(int(str_id))
torch.cuda.set_device(gpu_ids[0])

# Load Data
if opt.pmap:
    transform_train_list = [
            transforms.Resize((384,128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
else:
    transform_train_list = [
            transforms.Resize((384,128), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]

if opt.era > 0:
    transform_train_list = transform_train_list + [RandomErasing(probability = opt.era, mean=[0.0, 0.0, 0.0])]

data_transforms = {
    'train': transforms.Compose(transform_train_list)
}

if 'tri' not in opt.loss:
    if opt.pmap:
        cls_datasets = DukePartDataset(transform=data_transforms['train'])
    else:
        cls_datasets = DukeDataset(transform=data_transforms['train'])
    cls_loader = torch.utils.data.DataLoader(
        cls_datasets,
        batch_size=opt.batchsize,
        shuffle=True,
        num_workers=10,
        drop_last=True)
    dataset_sizes_allSample = len(cls_loader)
else:
    cls_datasets = DukeDataset(transform=data_transforms['train'])
    triplet_loader = torch.utils.data.DataLoader(
        triplet_datasets,
        sampler=RandomIdentitySampler(triplet_datasets, opt.num_instances),
        batch_size=opt.batchsize, num_workers=10,
        drop_last=True)
    dataset_sizes_metricSample = len(triplet_loader)

use_gpu = torch.cuda.is_available()


def test(model):
    model = model.eval()
    print('-' * 10)
    print('test model now...')
    data_transforms = transforms.Compose([
        transforms.Resize((384,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir = '/home/guojianyuan/Desktop/ReId/data/Duke_pytorch_eva'
    if opt.pmap:
        image_datasets = {x: DukePartDataset(mode=x, transform=data_transforms) for x in ['gallery','query']}
    else:
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir,x), data_transforms) for x in ['gallery','query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                             shuffle=False, num_workers=10) for x in ['gallery','query']}

    if opt.pmap:
        gallery_path = image_datasets['gallery'].name_list
        query_path = image_datasets['query'].name_list
        gallery_cam,gallery_label = get_id_with_part_map(gallery_path)
        query_cam,query_label = get_id_with_part_map(query_path)
    else:
        gallery_path = image_datasets['gallery'].imgs
        query_path = image_datasets['query'].imgs
        gallery_cam,gallery_label = get_id(gallery_path)
        query_cam,query_label = get_id(query_path)

    # Extract feature
    if opt.pmap:
        query_feature, query_feature_embed = extract_feature_with_part_map(model,dataloaders['query'])
        gallery_feature, gallery_feature_embed = extract_feature_with_part_map(model,dataloaders['gallery'])
    else:
        gallery_feature, gallery_feature_embed = extract_feature(model,dataloaders['gallery'])
        query_feature, query_feature_embed = extract_feature(model,dataloaders['query'])

    # Save to Matlab for check
    result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
    scipy.io.savemat('./'+opt.file_name+'/'+opt.name+'/'+opt.mat+'.mat',result)
    result = scipy.io.loadmat('./'+opt.file_name+'/'+opt.name+'/'+opt.mat+'.mat')

    query_feature = result['query_f']
    query_cam = result['query_cam'][0]
    query_label = result['query_label'][0]
    gallery_feature = result['gallery_f']
    gallery_cam = result['gallery_cam'][0]
    gallery_label = result['gallery_label'][0]

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    print('Pool5 top1:%f top5:%f top10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))

    result = {'gallery_f':gallery_feature_embed.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature_embed.numpy(),'query_label':query_label,'query_cam':query_cam}
    scipy.io.savemat('./'+opt.file_name+'/'+opt.name+'/'+opt.mat+'.mat',result)
    result = scipy.io.loadmat('./'+opt.file_name+'/'+opt.name+'/'+opt.mat+'.mat')

    query_feature = result['query_f']
    query_cam = result['query_cam'][0]
    query_label = result['query_label'][0]
    gallery_feature = result['gallery_f']
    gallery_cam = result['gallery_cam'][0]
    gallery_label = result['gallery_label'][0]

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0

    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    print('Embed top1:%f top5:%f top10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))


def train_model(model, optimizer, scheduler, num_epochs=25):
    start_time = time.time()

    for epoch in range(num_epochs):
        if epoch == 0:
            save_network(model, epoch)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                if 'tri' in opt.loss:
                    adjust_lr_triplet(optimizer, epoch)
                else:
                    adjust_lr_softmax(optimizer, epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_correct_category = 0

            # Iterate over data.
            if 'tri' in opt.loss:
                dataloaders = triplet_loader
            else:
                dataloaders = cls_loader
            for data in dataloaders:
                if opt.pmap:
                    inputs, labels, part_map, inputs_ori = data
                else:
                    inputs, labels = data          
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                    if opt.pmap:
                        part_map = Variable(part_map.cuda())
                        inputs_ori = Variable(inputs_ori.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                if opt.pmap:
                    [category, feature, pool5, embed] = model(inputs, part_map=part_map, path=inputs_ori)
                else:
                    [category, feature, pool5,_] = model(inputs)

                if not opt.PCB:
                    _,category_preds = torch.max(category.data, 1)

                    if opt.loss == 'softmax':
                        loss = criterion_softmax(category, labels)
                    elif opt.loss == 'labelsmooth': 
                        loss = criterion_labelsmooth(category, labels)
                    elif opt.loss == 'triplet':
                        loss,_,_ = criterion_triplet(feature, labels)
                    elif opt.loss == 'softmax+triplet':
                        loss_softmax = criterion_softmax(category, labels)
                        loss_triplet,_,_ = criterion_triplet(feature, labels)
                        loss = loss_softmax + loss_triplet
                    elif opt.loss == 'labelsmooth+triplet':
                        loss_softmax = criterion_labelsmooth(category, labels)
                        loss_triplet,_,_ = criterion_triplet(feature, labels)
                        loss = loss_softmax + loss_triplet

                else:
                    part = {}
                    sm = nn.Softmax(dim=1)
                    num_part = 6
                    for i in range(num_part):
                        part[i] = outputs[i]
                    score = sm(part[0]) + sm(part[1]) +sm(part[2]) + sm(part[3]) +sm(part[4]) +sm(part[5])
                    _, preds = torch.max(score.data, 1)
                    loss = criterion(part[0], labels)
                    for i in range(num_part-1):
                        loss += criterion(part[i+1], labels)
            
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()
                running_correct_category += torch.sum(category_preds == labels.data)

            if 'tri' not in opt.loss:
                epoch_loss = running_loss / dataset_sizes_allSample / opt.batchsize
                epoch_acc = running_correct_category.cpu().numpy() / dataset_sizes_allSample / opt.batchsize
            else:
                epoch_loss = running_loss / dataset_sizes_metricSample / opt.batchsize
                epoch_acc = running_correct_category.cpu().numpy() / dataset_sizes_metricSample / opt.batchsize

            print('{} Loss: {:.4f} Acc_category: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
          
            if 'tri' not in opt.loss:
                if epoch == 59 or epoch == 0:
                    save_network(model, epoch)
                    test(model)
            else:
                if epoch == 299 or epoch == 249 or epoch == 0:
                    save_network(model, epoch)
                    test(model)

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model


# Save model
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./' + opt.file_name, opt.name, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda(gpu_ids[0])


def load_network(network, path):
    pretrained_dict = torch.load(path)
    model_dict = network.state_dict()
    pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)
    return network

# Finetuning the convnet
model = P2Net(class_num=702, fusion=opt.fusion, layer=opt.layer, block_num=opt.block_num)

if use_gpu:
    cudnn.enabled = True
    cudnn.benchmark = True
    if len(gpu_ids)>1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids).cuda()
    else:
        model = model.cuda()

criterion_softmax = nn.CrossEntropyLoss()
criterion_triplet = tripletloss

# Train and evaluate
dir_name = os.path.join('./' + opt.file_name, opt.name)
if not os.path.isdir(dir_name):
    os.makedirs(dir_name)

# save opts
with open('%s/opts.json'%dir_name,'w') as fp:
    json.dump(vars(opt), fp, indent=1)

if len(opt.gpu_ids)>1:
    ignored_params = list(map(id, model.module.model.fc.parameters() )) + list(map(id, model.module.classifier.parameters() )) +\
                     list(map(id, model.module.context_l3_1.parameters() )) + list(map(id, model.module.context_l3_2.parameters() )) +\
                     list(map(id, model.module.context_l3_3.parameters() )) + list(map(id, model.module.context_l2_1.parameters() )) +\
                     list(map(id, model.module.context_l2_2.parameters() ))
    base_params = filter(lambda p: id(p) not in ignored_params, model.module.parameters() )
    optimizer = torch.optim.SGD([
                     {'params': base_params},
                     {'params': model.module.model.fc.parameters()},
                     {'params': model.module.classifier.parameters()},
                     {'params': model.module.context_l3_1.parameters()},
                     {'params': model.module.context_l3_2.parameters()},
                     {'params': model.module.context_l3_3.parameters()},
                     {'params': model.module.context_l2_1.parameters()},
                     {'params': model.module.context_l2_2.parameters()}
                     ], lr=0.1, weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)
else:
    ignored_params = list(map(id, model.model.fc.parameters() )) + list(map(id, model.classifier.parameters() )) +\
                     list(map(id, model.context_l3_1.parameters() )) + list(map(id, model.context_l3_2.parameters() )) +\
                     list(map(id, model.context_l3_3.parameters() )) + list(map(id, model.context_l2_1.parameters() )) +\
                     list(map(id, model.context_l2_2.parameters() ))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters() )
    optimizer = torch.optim.SGD([
                     {'params': base_params},
                     {'params': model.model.fc.parameters()},
                     {'params': model.classifier.parameters()},
                     {'params': model.context_l3_1.parameters()},
                     {'params': model.context_l3_2.parameters()},
                     {'params': model.context_l3_3.parameters()},
                     {'params': model.context_l2_1.parameters()},
                     {'params': model.context_l2_2.parameters()}
                     ], lr=0.1, weight_decay=opt.weight_decay, momentum=0.9, nesterov=True)


def adjust_lr_triplet(optimizer, ep):
    if ep < 20:
        lr = 1e-2 * (ep + 1) / 2
    elif ep < 130:
        lr = 1e-1
    elif ep < 200:
        lr = 1e-2
    elif ep < 240:
        lr = 1e-3
    elif ep < 280:
        lr = 1e-3 * (ep - 240 + 1) / 40
    elif ep < 340:
        lr = 1e-3
    for index in range(len(optimizer.param_groups)):
        if index == 0:
            optimizer.param_groups[index]['lr'] = lr * 0.1
        else:
            optimizer.param_groups[index]['lr'] = lr


def adjust_lr_softmax(optimizer, ep):
    if ep < 40:
        lr = 0.1
    elif ep < 60:
        lr = 0.01
    else:
        lr = 0.001
    for index in range(len(optimizer.param_groups)):
        if index == 0:
            optimizer.param_groups[index]['lr'] = lr * 0.1
        else:
            optimizer.param_groups[index]['lr'] = lr


model = train_model(model, optimizer, None, num_epochs=opt.epoch)