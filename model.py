import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import pdb
import ResNet

from blocks import MSPyramidAttentionContextModule


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block_1 = []
        add_block_1 += [nn.Linear(input_dim, num_bottleneck)] 
        add_block_1 += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block_1 += [nn.LeakyReLU(0.5)]
            #add_block_1 += [nn.SELU()]
        if dropout:
            add_block_1 += [nn.Dropout(p=0.3)]
        add_block_1 = nn.Sequential(*add_block_1)
        add_block_1.apply(weights_init_kaiming)
        
        classifier_1 = []
        classifier_1 += [nn.Linear(num_bottleneck, class_num)]
        classifier_1 = nn.Sequential(*classifier_1)
        classifier_1.apply(weights_init_classifier)

        self.add_block_1 = add_block_1
        self.classifier_1 = classifier_1

    def forward(self, x):
        x = self.add_block_1(x)
        x = self.classifier_1(x)

        return x


class P2Net(nn.Module):
    def __init__(self, class_num=702, fusion='+', layer='50', block_num=0):
        super().__init__()
        if layer == '50':
            backbone = ResNet.resnet50(pretrained =True)
        elif layer == '152':
            backbone = ResNet.resnet152(pretrained=True)
        
        # avg pooling to global pooling
        backbone.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.model = backbone
        self.classifier = ClassBlock(2048, class_num, dropout=False, relu=True, num_bottleneck=256)
        
        self.context_l2_1 = MSPyramidAttentionContextModule(in_channels=1024, out_channels=1024, c1=512, c2=512, 
            dropout=0, fusion=fusion, sizes=([1]), if_gc=0) if block_num > 0 else nn.Sequential()
        '''
        self.context_l2_2 = MSPyramidAttentionContextModule(in_channels=2048, out_channels=2048, c1=1024, c2=1024, 
            dropout=0, fusion='+', sizes=([1]), if_gc=0)
        self.context_l3_1 = MSPyramidAttentionContextModule(in_channels=512, out_channels=512, c1=256, c2=256, 
            dropout=0, fusion='+', sizes=([1]), if_gc=0)
        self.context_l3_2 = MSPyramidAttentionContextModule(in_channels=512, out_channels=512, c1=256, c2=256, 
            dropout=0, fusion='+', sizes=([1]), if_gc=0)
        self.context_l3_3 = MSPyramidAttentionContextModule(in_channels=1024, out_channels=1024, c1=512, c2=512, 
            dropout=0, fusion='+', sizes=([1]), if_gc=0)
        '''
        self.context_l2_2 = nn.Sequential()
        self.context_l3_1 = nn.Sequential()
        self.context_l3_2 = nn.Sequential()
        self.context_l3_3 = nn.Sequential()
        
    def forward(self, x, part_map=None, path=None):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)

        x = self.model.layer2(x)

        x = self.model.layer3(x)
        x = self.context_l2_1(x, part_map)

        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        feature = self.classifier.add_block_1(x)
        category = self.classifier.classifier_1(feature)

        return category, feature, x, x
     

if __name__ == '__main__':
    net = P2Net(751)
    input = Variable(torch.FloatTensor(8,3,256,128).cuda())
    net=net.cuda()
    output = net(input)