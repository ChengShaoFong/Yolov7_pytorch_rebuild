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
from utils.initParam import *

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

    def forward(self, x : Tensor, accumScale=1.0, isLayer1=False) -> Tensor:
        tmpList = []
        x, weight, xScale, xZeroPt, convScale = inputQuantizer(x, self.conv, accumScale, isLayer1=isLayer1, types="conv")

        x = self.conv(x)
        x = floatConverter(x, xScale, self.conv)
        
        x = self.relu(x)
        x, accumScale, accumZeroPt = accumQuantizer(x, isOffsetReset=self.isOffsetReset)
        tmpList.append((float(xScale), int(xZeroPt), float(accumScale), int(accumZeroPt), float(convScale)))
        
        return x, accumScale, accumZeroPt, tmpList

class fcLayer(PruningModule):
    def __init__(self, in_feature_num, out_feature_num, mask=True):
        super(fcLayer, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        # linear = nn.Linear
        self.fc = linear(in_feature_num, out_feature_num)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x:Tensor, accumScale=1.0) -> Tensor:
        # x, weight, xScale, xZeroPt, convScale = inputQuantizer(x, self.fc, accumScale, types="fc")
        x = self.fc(x)
        # x = floatConverter(x, xScale, self.fc)
        x = self.relu(x)
        # x, accumScale, accumZeroPt = accumQuantizer(x, isOffsetReset=self.isOffsetReset)
        # tmpList.append((float(xScale), int(xZeroPt), float(accumScale), int(accumZeroPt), float(convScale)))
        # return x, accumScale, accumZeroPt, tmpList
        return x

class VGG16_net(PruningModule):
    def __init__(self, dataset, in_channels=3, num_classes=1000, feature_num=4096):
        super(VGG16_net, self).__init__()

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

    def forward(self, x:Tensor, idx) -> Tensor:
        cfgFilePath = "./cfg/" + self.version + "/vgg16/" + self.dataset + "/" + str(w_quantBitwidth) + "-bit/" +  self.dataset + "_vgg16_trainObserver" + str(idx) + ".cfg"
        cfgFile = open(cfgFilePath, "w+")
        quantList = []

        # initial step set accumScale to 1.0
        x, accumScale, accumZeroPt, tmpList = self.conv1(x, accumScale=1.0, isLayer1=True)
        quantList.append(tmpList[0])
        x, accumScale, accumZeroPt, tmpList = self.conv2(x, accumScale)
        quantList.append(tmpList[0])
        x = x.type(torch.float32)
        x = self.maxpool(x)
        # x = x.type(torch.int32)

        x, accumScale, accumZeroPt, tmpList = self.conv3(x, accumScale)
        quantList.append(tmpList[0])
        x, accumScale, accumZeroPt, tmpList = self.conv4(x, accumScale)
        quantList.append(tmpList[0])
        x = x.type(torch.float32)
        x = self.maxpool(x)
        # x = x.type(torch.int32)

        x, accumScale, accumZeroPt, tmpList = self.conv5(x, accumScale)
        quantList.append(tmpList[0])
        x, accumScale, accumZeroPt, tmpList = self.conv6(x, accumScale)
        quantList.append(tmpList[0])
        x, accumScale, accumZeroPt, tmpList = self.conv7(x, accumScale)
        quantList.append(tmpList[0])
        x = x.type(torch.float32)
        x = self.maxpool(x)
        # x = x.type(torch.int32)

        x, accumScale, accumZeroPt, tmpList = self.conv8(x, accumScale)
        quantList.append(tmpList[0])
        x, accumScale, accumZeroPt, tmpList = self.conv9(x, accumScale)
        quantList.append(tmpList[0])
        x, accumScale, accumZeroPt, tmpList = self.conv10(x, accumScale)
        quantList.append(tmpList[0])
        x = x.type(torch.float32)
        x = self.maxpool(x)
        # x = x.type(torch.int32)

        x, accumScale, accumZeroPt, tmpList = self.conv11(x, accumScale)
        quantList.append(tmpList[0])
        x, accumScale, accumZeroPt, tmpList = self.conv12(x, accumScale)
        quantList.append(tmpList[0])
        x, accumScale, accumZeroPt, tmpList = self.conv13(x, accumScale)
        quantList.append(tmpList[0])
        x = x.type(torch.float32)
        x = self.maxpool(x)
        # x = x.type(torch.int32)
        
        x = deQuantizer(x, accumScale, accumZeroPt)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x, accumScale, accumZeroPt, tmpList = self.fc1(x, accumScale)
        # quantList.append(tmpList[0])
        x = self.dropout(x)
        x = self.fc2(x)
        # x, accumScale, accumZeroPt, tmpList = self.fc2(x, accumScale)
        # quantList.append(tmpList[0])
        x = self.dropout(x)
        x = self.fc3(x)
        # x, accumScale, accumZeroPt, tmpList = self.fc3(x, accumScale)
        # quantList.append(tmpList[0])
  
        for item in quantList:
            cfgFile.write(str(item) + '\n')
        cfgFile.close()

        return x
        

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
    if types == "conv":
        weightScale = layerObj.conv.scale
        convScale = xScale * weightScale
        x = x.type(torch.float32)
        weight = layerObj.conv.weight.data.type(torch.float32)
       
    elif types == "fc":
        weightScale = layerObj.scale
        convScale = xScale * weightScale
        x = x.type(torch.float32)
        weight = layerObj.weight.data.type(torch.float32)
       
    return x, weight, xScale, xZeroPt, convScale

def floatConverter(x: Tensor, inputScale, layerObj):
    weightScale = layerObj.conv.scale
    layerScale = inputScale * weightScale
    out_bias = layerObj.conv.bias.data.view(1, -1, 1, 1)
    out_weightx = x - out_bias
    biasAdd = layerObj.conv.biasFloat.data
    # output = out_weightx * layerScale / (f_compensateScale ** 2) + biasAdd.view(1, -1, 1, 1)
    output = out_weightx * layerScale / (f_compensateScale ** w_compensateScale) + biasAdd.view(1, -1, 1, 1)
    return output

def accumQuantizer(x: Tensor, isOffsetReset=True):
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
