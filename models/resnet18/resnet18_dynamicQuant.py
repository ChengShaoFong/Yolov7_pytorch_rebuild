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
        conv2d = MaskedConv2d if mask else nn.Conv2d
        # conv2d = nn.Conv2d
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
        if stride != (1,1) or in_channels != out_channels:
            self.check = True
            self.shortcut = nn.Sequential(OrderedDict([
                ('conv', conv2d(in_channels, out_channels, kernel_size=(1,1), stride=stride, padding=(0,0), bias=False)),
                ('batchnorm', nn.BatchNorm2d(out_channels)),
            ]))
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x : Tensor, accumScale) -> Tensor:
        tmpList = []
    
        # scout = x * accumScale

        out, xScale, xZeroPt = inputQuantizer(x, self.conv1, accumScale)
        xq, xqScale = out, xScale

        scout = x * xScale
       
        out = self.conv1(out)
        out = floatConverter(out, xScale, self.conv1)
        out = self.relu1(out)
        out, accumScale, accumZeroPt = accumQuantizer(out, calType="conv", isOffsetReset=self.isOffsetReset)
        tmpList.append((float(xScale), int(xZeroPt), float(accumScale), int(accumZeroPt)))

        out, xScale, xZeroPt = inputQuantizer(out, self.conv2, accumScale)
        out = self.conv2(out)
        out = floatConverter(out, xScale, self.conv2)

        if self.check == True:
            # scout = shortcut output
            scout = self.shortcut(xq)
            # Add computation
            scout = floatConverter(scout, xqScale, self.shortcut)
        
        out = out + scout
        out = self.relu2(out)
        
        out, accumScale, accumZeroPt = accumQuantizer(out, calType="conv", isOffsetReset=self.isOffsetReset)
        tmpList.append((float(xScale), int(xZeroPt), float(accumScale), int(accumZeroPt)))
        return out, accumScale, accumZeroPt, tmpList

class ResNet18(PruningModule):
    # For CIFAR10, the "num_classes" should be set to 10.
    # For ImageNet, the "num_classes" should be set to 1000.
    def __init__(self, ResBlock, dataset, num_classes=10, feature_num=512, mask=True):
        super(ResNet18, self).__init__()
        # linear = MaskedLinear if mask else nn.Linear
        linear = nn.Linear
        conv2d = MaskedConv2d if mask else nn.Conv2d
        self.version = ""
        
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
        self.fc = linear(feature_num * mulScale, num_classes)
        self.avgpool2d = nn.AvgPool2d(4) # square window of (kernel) size = 4

    def forward(self, x : Tensor, idx) -> Tensor:
        cfgFilePath = "./cfg/" + self.version + "/resnet18/" + self.dataset + "/" + str(w_quantBitwidth) + "-bit/" +  self.dataset + "_resnet18_trainObserver" + str(idx) + ".cfg"
        cfgFile = open(cfgFilePath, "w+")
        quantList = []
        
        x, xScale, xZeroPt = inputQuantizer(x, self.inputConv, isLayer1=True)  
        out = self.inputConv(x)
        out = floatConverter(out, xScale, self.inputConv)  #float -> int
        out = self.inReLU(out)
        out, accumScale, accumZeroPt = accumQuantizer(out, calType="conv", isOffsetReset=True)  # xq  
        quantList.append((float(xScale), int(xZeroPt), float(accumScale), int(accumZeroPt)))
        
        out, accumScale, accumZeroPt, tmpList = self.Res1(out, accumScale)
        quantList.append(tmpList[0]) # store the quantization info of 'conv1' in ResBlock
        quantList.append(tmpList[1]) # store the quantization info of 'conv2' in ResBlock

        out, accumScale, accumZeroPt, tmpList = self.Res2(out, accumScale)
        quantList.append(tmpList[0])
        quantList.append(tmpList[1])

        out, accumScale, accumZeroPt, tmpList = self.Res3(out, accumScale)
        quantList.append(tmpList[0])
        quantList.append(tmpList[1])

        out, accumScale, accumZeroPt, tmpList = self.Res4(out, accumScale)
        quantList.append(tmpList[0])
        quantList.append(tmpList[1])

        out, accumScale, accumZeroPt, tmpList = self.Res5(out, accumScale)
        quantList.append(tmpList[0])
        quantList.append(tmpList[1])

        out, accumScale, accumZeroPt, tmpList = self.Res6(out, accumScale)
        quantList.append(tmpList[0])
        quantList.append(tmpList[1])

        out, accumScale, accumZeroPt, tmpList = self.Res7(out, accumScale)
        quantList.append(tmpList[0])
        quantList.append(tmpList[1])
        
        out, accumScale, accumZeroPt, tmpList = self.Res8(out, accumScale)
        quantList.append(tmpList[0])
        quantList.append(tmpList[1])

        out = deQuantizer(out, accumScale, accumZeroPt)

        out = self.avgpool2d(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        for item in quantList:
            cfgFile.write(str(item) + '\n')
        cfgFile.close()

        return out

# Quantization operations
def getValueMinMax(x: Tensor) -> Tensor:
    valueMin = torch.min(x)
    valueMax = torch.max(x)
    return valueMin, valueMax

def inputQuantizer(x: Tensor, layerObj, accumScale=1.0, isLayer1=False, types=None):
    if (isLayer1):
        inputMin, inputMax = getValueMinMax(x)
        quantizer = quantization(inputMin, inputMax, bits=f_quantBitwidth, isAsymmetric=True, isClipping=False)
        x, xScale, xZeroPt = quantizer.valueMapping(x, f_compensateScale)
        # Do "input's" offset reset
        x = quantizer.offsetReset(x, xZeroPt)
        
    else:
        xScale = accumScale
        xZeroPt = 0
    
    # Store the scale and set the calculation type to "torch.int32"
    # weightScale = layerObj.conv.scale
    # convScale = xScale * weightScale
    x = x.type(torch.float32)
    # weight = layerObj.conv.weight.data.type(torch.int32)

    return x, xScale, xZeroPt

def floatConverter(x: Tensor, inputScale, layerObj): # float -> int
    weightScale = layerObj.conv.scale
    layerScale = inputScale * weightScale
    # biasAdd = layerObj.conv.bias.data.view(1, -1, 1, 1) * (weightScale - layerScale)
    out_bias = layerObj.conv.bias.data.view(1, -1, 1, 1)
    out_weightx = x - out_bias
    # biasAdd = out_bias * weightScale
    biasAdd = layerObj.conv.biasFloat.data
    # output = out_weightx * layerScale / (compensateScale ** 2) + biasAdd.view(1, -1, 1, 1)
    output = out_weightx * layerScale / (f_compensateScale * w_compensateScale) + biasAdd.view(1, -1, 1, 1)
    
    return output

def accumQuantizer(x: Tensor, calType="conv", isOffsetReset=True):
    accumMin, accumMax = getValueMinMax(x)
    quantizer = quantization(accumMin, accumMax, bits=f_quantBitwidth, isAsymmetric=True, isClipping=False)
    xq, accumScale, accumZeroPt = quantizer.valueMapping(x, f_compensateScale)
    if isOffsetReset:
        xq = quantizer.offsetReset(xq, accumZeroPt)
        accumZeroPt = 0
    xq = xq.type(torch.float32)
    return xq, accumScale, accumZeroPt

def deQuantizer(x: Tensor, accumScale, accumZeroPt):
    x = (x - accumZeroPt) * accumScale    
    return x