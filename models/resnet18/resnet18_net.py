import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from collections import OrderedDict
from prune.pruning import *

class ResBlock(PruningModule):
    def __init__(self, in_channels, out_channels, stride=(1,1), mask=True):
        super(ResBlock, self).__init__()
        conv2d = MaskedConv2d if mask else nn.Conv2d
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', conv2d(in_channels, out_channels, kernel_size=(3,3), stride=stride, padding=(1,1), bias=True)),
            ('batchnorm', nn.BatchNorm2d(out_channels)),
        ]))
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Sequential(OrderedDict([
            ('conv',conv2d(out_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True)),
            ('batchnorm', nn.BatchNorm2d(out_channels))
        ]))
        self.shortcut = nn.Sequential()
        self.check = False # To place shortcut or not
        if stride != (1,1) or in_channels != out_channels:
            self.check = True
            self.shortcut = nn.Sequential(OrderedDict([
                ('conv', conv2d(in_channels, out_channels, kernel_size=(1,1), stride=stride, padding=(0,0), bias=True)),
                ('batchnorm', nn.BatchNorm2d(out_channels))
            ]))
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        if self.check == True:
            out = out + self.shortcut(x)
        out = self.relu2(out)
        return out

class ResNet18(PruningModule):
    # For CIFAR10, the "num_classes" should be set to 10.
    # For ImageNet, the "num_classes" should be set to 1000.
    def __init__(self, ResBlock, dataset="cifar10", num_classes=10, feature_num=512, mask=True):
        super(ResNet18, self).__init__()
        # linear = MaskedLinear if mask else nn.Linear
        linear = nn.Linear
        conv2d = MaskedConv2d if mask else nn.Conv2d
        
        if dataset == "cifar10" or dataset == "cifar100":
            self.input_channels = 3
            mulScale = 1
        elif dataset == "imagenet-tiny":
            self.input_channels = 3
            mulScale = 49 # 224x224 -> 7x7
        elif dataset == "mnist":
            self.input_channels = 1
            mulScale = 1
        
        # Input layer
        self.inputConv = nn.Sequential(OrderedDict([
            ('conv', conv2d(self.input_channels, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True)), 
            ('batchnorm', nn.BatchNorm2d(64)),
        ]))
        self.inReLU = nn.ReLU(inplace=True)
        
        # ResBlocks : Each one contains 2 convolution layers.
        self.Res1 = ResBlock(64, 64, stride=(1,1))
        self.Res2 = ResBlock(64, 64, stride=(1,1))
        self.Res3 = ResBlock(64, 128, stride=(2,2))
        self.Res4 = ResBlock(128, 128, stride=(1,1))
        self.Res5 = ResBlock(128, 256, stride=(2,2))
        self.Res6 = ResBlock(256, 256, stride=(1,1))
        self.Res7 = ResBlock(256, 512, stride=(2,2))
        self.Res8 = ResBlock(512, 512, stride=(1,1))
        
        # Fully-connected and AvgPool2d
        self.fc = linear(feature_num * mulScale, num_classes)
        self.avgpool2d = nn.AvgPool2d(4) # square window of (kernel) size = 4

    def forward(self, x):
        out = self.inputConv(x)
        out = self.inReLU(out)
        
        out = self.Res1(out)
        out = self.Res2(out)
        out = self.Res3(out)
        out = self.Res4(out)
        out = self.Res5(out)
        out = self.Res6(out)
        out = self.Res7(out)
        out = self.Res8(out)

        out = self.avgpool2d(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
        