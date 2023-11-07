import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import Tensor
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from collections import OrderedDict
from prune.pruning import *
from prune.quantization import *

class convLayer(PruningModule):
    def __init__(self, in_channels, out_channels, mask=True):
        super(convLayer, self).__init__()
        conv2d = MaskedConv2d if mask else nn.Conv2d
        self.conv = nn.Sequential(OrderedDict([
            ('conv', conv2d(in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True)),
            ('batchnorm2d', nn.BatchNorm2d(out_channels)),
        ]))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class fcLayer(PruningModule):
    def __init__(self, in_feature_num, out_feature_num, mask=True):
        super(fcLayer, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.fc = linear(in_feature_num, out_feature_num)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x

class VGG16_net(PruningModule):
    def __init__(self, dataset="cifar10", in_channels=3, num_classes=1000, feature_num=4096, mask=True):
        super(VGG16_net, self).__init__()

        self.dataset = dataset
        if dataset == "cifar10" or dataset == "cifar100":
            self.in_channels = 3
            mulScale = 1
        elif dataset == "imagenet-tiny":
            self.in_channels = 3
            mulScale = 49 # 224x224 -> 7x7
        elif dataset == "mnist":
            self.in_channels = 1
            mulScale = 1

        self.conv1 = convLayer(in_channels, 64)
        self.conv2 = convLayer(64, 64)
        self.conv3 = convLayer(64, 128)
        self.conv4 = convLayer(128, 128)
        self.conv5 = convLayer(128, 256)
        self.conv6 = convLayer(256, 256)
        self.conv7 = convLayer(256, 512)
        self.conv8 = convLayer(512, 512)
        self.conv9 = convLayer(512, 512)
        self.conv10 = convLayer(512, 512)
        self.conv11 = convLayer(512, 512)
        self.conv12 = convLayer(512, 512)
        self.conv13 = convLayer(512, 512)

        self.fc1 = fcLayer(512*mulScale, feature_num)
        self.fc2 = fcLayer(feature_num, feature_num)
        self.fc3 = fcLayer(feature_num, num_classes)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.maxpool(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.maxpool(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x