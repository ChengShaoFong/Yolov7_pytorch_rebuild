import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import Tensor
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from collections import OrderedDict
from prune.pruning_uint8 import *
from prune.quantization import *

class convLayer(PruningModule):
    def __init__(self, in_channels, out_channels, isOffsetReset=True, mask=True):
        super(convLayer, self).__init__()
        conv2d = MaskedConv2d if mask else nn.Conv2d

        self.isOffsetReset = isOffsetReset
        self.conv = nn.Sequential(OrderedDict([
            ('conv', conv2d(in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=True)),
            ('batchnorm2d', nn.BatchNorm2d(out_channels)),
        ]))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x:Tensor, cfgFile=None, cfgidx=0, isLayer1=False) -> Tensor:
        out, xScale, xZeroPt = inputQuantizer(x, cfgFile=cfgFile, cfgidx=cfgidx, isLayer1=isLayer1)

        
        out = self.conv(out) 
        out = floatConverter(out, xScale, self.conv)
        out = self.relu(out)
        out, accumScale, accumZeroPt = accumQuantizer(out, isOffsetReset=self.isOffsetReset, cfgFile=cfgFile, cfgidx=cfgidx)

        return out, accumScale, accumZeroPt

class fcLayer(PruningModule):
    def __init__(self, in_feature_num, out_feature_num, mask=False):
        super(fcLayer, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.fc = linear(in_feature_num, out_feature_num)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x

class VGG16_net(PruningModule):
    def __init__(self, dataset="cifar10", in_channels=3, num_classes=1000, feature_num=4096):
        super(VGG16_net, self).__init__()

        self.dataset = dataset
        if dataset == "cifar10":
            input_channels = 3
            mulScale = 1
            num_classes = 10
        elif dataset == "cifar100":
            input_channels = 3
            mulScale = 1
            num_classes = 100
        elif dataset == "imagenet-tiny":
            input_channels = 3
            mulScale = 49 # 224x224 -> 7x7
            num_classes = 200
        elif dataset == "mnist":
            input_channels = 1
            mulScale = 1
            num_classes = 10

        self.conv1 = convLayer(input_channels, 64)
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
    

    def forward(self, x:Tensor) -> Tensor:

        x, accumScale, accumZeroPt = self.conv1(x, cfgFile=self.cfgFile, cfgidx=0, isLayer1=True)
        x, accumScale, accumZeroPt = self.conv2(x, cfgFile=self.cfgFile, cfgidx=1)
        x = x.type(torch.float32)
        x = self.maxpool(x)
        # x = x.type(torch.int32)

        x, accumScale, accumZeroPt = self.conv3(x, cfgFile=self.cfgFile, cfgidx=2)
        x, accumScale, accumZeroPt = self.conv4(x, cfgFile=self.cfgFile, cfgidx=3)
        x = x.type(torch.float32)
        x = self.maxpool(x)
        # x = x.type(torch.int32)

        x, accumScale, accumZeroPt = self.conv5(x, cfgFile=self.cfgFile, cfgidx=4)
        x, accumScale, accumZeroPt = self.conv6(x, cfgFile=self.cfgFile, cfgidx=5)
        x, accumScale, accumZeroPt = self.conv7(x, cfgFile=self.cfgFile, cfgidx=6)
        x = x.type(torch.float32)
        x = self.maxpool(x)
        # x = x.type(torch.int32)

        x, accumScale, accumZeroPt = self.conv8(x, cfgFile=self.cfgFile, cfgidx=7)
        x, accumScale, accumZeroPt = self.conv9(x, cfgFile=self.cfgFile, cfgidx=8)
        x, accumScale, accumZeroPt = self.conv10(x, cfgFile=self.cfgFile, cfgidx=9)
        x = x.type(torch.float32)
        x = self.maxpool(x)
        # x = x.type(torch.int32)

        x, accumScale, accumZeroPt = self.conv11(x, cfgFile=self.cfgFile, cfgidx=10)
        x, accumScale, accumZeroPt = self.conv12(x, cfgFile=self.cfgFile, cfgidx=11)
        x, accumScale, accumZeroPt = self.conv13(x, cfgFile=self.cfgFile, cfgidx=12)
        x = x.type(torch.float32)
        x = self.maxpool(x)
        # x = x.type(torch.int32)

        x = deQuantizer(x, accumScale, accumZeroPt)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x
        

# Quantization operations
def getValueMinMax(x: Tensor) -> Tensor:
    valueMin = torch.min(x)
    valueMax = torch.max(x)
    return valueMin, valueMax

def inputQuantizer(x: Tensor, cfgFile=None, cfgidx=0, isLayer1=False):
    xScale = cfgFile[cfgidx][0]
    xZeroPt = cfgFile[cfgidx][1]
    
    if (isLayer1):
        x = torch.round(x / xScale * f_compensateScale)

    x = x.type(torch.float32)
    return x, xScale, xZeroPt

def floatConverter(x: Tensor, inputScale, layerObj):
    weightScale = layerObj.conv.scale
    layerScale = inputScale * weightScale
    out_bias = layerObj.conv.bias.data.view(1, -1, 1, 1)
    out_weightx = x - out_bias
    biasAdd = layerObj.conv.biasFloat.data.view(1, -1, 1, 1)
    # output = out_weightx * layerScale / (compensateScale ** 2) + biasAdd
    output = out_weightx * layerScale / (f_compensateScale * w_compensateScale) + biasAdd
    return output

def accumQuantizer(x: Tensor, isOffsetReset=True, cfgFile=None, cfgidx=0):
    accumScale = cfgFile[cfgidx][2]
    accumZeroPt = cfgFile[cfgidx][3]
    xq = torch.round(x / accumScale * f_compensateScale) + accumZeroPt
    if isOffsetReset:
        xq = xq - accumZeroPt
        accumZeroPt = 0
    xq = xq.type(torch.float32)
    return xq, accumScale, accumZeroPt

def deQuantizer(x: Tensor, accumScale, accumZeroPt):
    x = (x - accumZeroPt) * accumScale
    return x