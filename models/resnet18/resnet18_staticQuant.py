import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from collections import OrderedDict
from prune.pruning_uint8 import *
from prune.quantization import *
from utils.initParam import *

class ResBlock(PruningModule):
    def __init__(self, in_channels, out_channels, stride=(1,1), mask=True, accumScale=1.0, isOffsetReset=True):
        super(ResBlock, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        conv2d = MaskedConv2d if mask else nn.Conv2d
        self.isOffsetReset = isOffsetReset
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', conv2d(in_channels, out_channels, kernel_size=(3,3), stride=stride, padding=(1,1), bias=False)),
            ('batchnorm', nn.BatchNorm2d(out_channels)),
        ]))
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Sequential(OrderedDict([
            ('conv',conv2d(out_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)),
            ('batchnorm', nn.BatchNorm2d(out_channels)),
        ]))
        self.shortcut = nn.Sequential()
        self.check = False # To place shortcut or not
        if  stride != (1,1) or in_channels != out_channels: #stride != (1,1) or
            self.check = True
            self.shortcut = nn.Sequential(OrderedDict([
                ('conv', conv2d(in_channels, out_channels, kernel_size=(1,1), stride=stride, padding=(0,0), bias=False)),
                ('batchnorm', nn.BatchNorm2d(out_channels)),
            ]))
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x : Tensor, cfgFile=None, cfgidx=0) -> Tensor:
        out, xScale, xZeroPt = inputQuantizer(x, cfgFile=cfgFile, cfgidx=cfgidx)
        xq, xqScale = out, xScale
        scout = x * xScale   #????
        
        out = self.conv1(out)
        out = floatConverter(out, xScale, self.conv1)
        out = self.relu1(out)
        out, accumScale, accumZeroPt = accumQuantizer(out, isOffsetReset=self.isOffsetReset, cfgFile=cfgFile, cfgidx=cfgidx)

        out, xScale, xZeroPt = inputQuantizer(out, cfgFile=cfgFile, cfgidx=cfgidx+1)
        out = self.conv2(out)
        out = floatConverter(out, xScale, self.conv2)

        if self.check == True:
            # scout = shortcut output
            scout = self.shortcut(xq)
            # Add computation
            scout = floatConverter(scout, xqScale, self.shortcut)

        out = out + scout

        out = self.relu2(out)
        out, accumScale, accumZeroPt = accumQuantizer(out, isOffsetReset=self.isOffsetReset, cfgFile=cfgFile, cfgidx=cfgidx+1)
        return out, accumScale, accumZeroPt

class ResNet18(PruningModule):
    # For CIFAR10, the "num_classes" should be set to 10.
    # For ImageNet, the "num_classes" should be set to 1000.
    def __init__(self, ResBlock, dataset, num_classes=10, feature_num=512, mask=True):
        super(ResNet18, self).__init__()
        # linear = MaskedLinear if mask else nn.Linear
        linear = nn.Linear
        conv2d = MaskedConv2d if mask else nn.Conv2d

        self.dataset = dataset
        if dataset == "cifar10":
            self.input_channels = 3
            mulScale = 1
            num_classes = 10
        elif dataset == "cifar100":
            self.input_channels = 3
            mulScale = 1
            num_classes = 100
        elif dataset == "imagenet-tiny":
            self.input_channels = 3
            mulScale = 49 # 224x224 -> 7x7
            num_classes = 200
        elif dataset == "mnist":
            self.input_channels = 1
            mulScale = 1
            num_classes = 10
        
        self.cfgFile = []
        
        # Input layer
        self.inputConv = nn.Sequential(OrderedDict([
            ('conv', conv2d(self.input_channels, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)), 
            ('batchnorm', nn.BatchNorm2d(64)),
        ]))
        self.inReLU = nn.ReLU(inplace=True)
        
        # ResBlocks : Each one contains 2 convolution layers.
        self.Res1 = ResBlock(64, 64, stride=(1,1), mask=True)
        self.Res2 = ResBlock(64, 64, stride=(1,1), mask=True)
        self.Res3 = ResBlock(64, 128, stride=(2,2), mask=True)
        self.Res4 = ResBlock(128, 128, stride=(1,1), mask=True)
        self.Res5 = ResBlock(128, 256, stride=(2,2), mask=True)
        self.Res6 = ResBlock(256, 256, stride=(1,1), mask=True)
        self.Res7 = ResBlock(256, 512, stride=(2,2), mask=True)
        self.Res8 = ResBlock(512, 512, stride=(1,1), mask=True, isOffsetReset=False)
        
        # Fully-connected and AvgPool2d
        self.avgpool2d = nn.AvgPool2d(4) # square window of (kernel) size = 4
        self.fc = linear(feature_num * mulScale, num_classes)
        

    def forward(self, x : Tensor) -> Tensor:
        
        x, xScale, xZeroPt = inputQuantizer(x, cfgFile=self.cfgFile, cfgidx=0, isLayer1=True)
        out = self.inputConv(x)
        out = floatConverter(out, xScale, self.inputConv)
        out = self.inReLU(out)

        
        out, accumScale, accumZeroPt = accumQuantizer(out, isOffsetReset=True, cfgFile=self.cfgFile, cfgidx=0)
        out, accumScale, accumZeroPt = self.Res1(out, cfgFile=self.cfgFile, cfgidx=1)
        out, accumScale, accumZeroPt = self.Res2(out, cfgFile=self.cfgFile, cfgidx=3)
        out, accumScale, accumZeroPt = self.Res3(out, cfgFile=self.cfgFile, cfgidx=5)
        out, accumScale, accumZeroPt = self.Res4(out, cfgFile=self.cfgFile, cfgidx=7)
        out, accumScale, accumZeroPt = self.Res5(out, cfgFile=self.cfgFile, cfgidx=9)
        out, accumScale, accumZeroPt = self.Res6(out, cfgFile=self.cfgFile, cfgidx=11)
        out, accumScale, accumZeroPt = self.Res7(out, cfgFile=self.cfgFile, cfgidx=13)
        out, accumScale, accumZeroPt = self.Res8(out, cfgFile=self.cfgFile, cfgidx=15)
        
       
        out = deQuantizer(out, accumScale, accumZeroPt)
        
        out = self.avgpool2d(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

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