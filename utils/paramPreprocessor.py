import copy
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from prune.quantization import *
from utils.utils import *

class DummyLayer(torch.nn.Module):
    # Use for the "nn.Identity()" replacement.
    # To avoid extra computation from "nn.Identity()".
    def __init__(self):
        super(DummyLayer, self).__init__()

    def forward(self, x):
        return x

def getMinMax(x: Tensor) -> Tensor:
    minimum = torch.min(x)
    maximum = torch.max(x)
    return minimum, maximum

def getQuantInfo(weight, bias, quantBitwidth):
    # Get weight and bias range
    
    w_rangeMin, w_rangeMax = getMinMax(weight.data)
    b_rangeMin, b_rangeMax = getMinMax(bias.data)
    
    # check total range
    rangeMin = min(w_rangeMin, b_rangeMin)
    rangeMax = max(w_rangeMax, b_rangeMax)
    
    # Quantization
    quantizer = quantization(rangeMin, rangeMax, bits=quantBitwidth, isAsymmetric=True, isClipping=True)
    compScale = 1 << quantBitwidth # following weight or feature map compensate scale
    
    qInt, qScale, qZeroPt = quantizer.valueMapping(weight.data, compScale)
    

    return qScale, qZeroPt

def weightQuant(weight, qScale, qZeroPt):
    # The value "compensateScale" is defined in "prune.quantization"
    weightInt = torch.round(w_compensateScale * weight / qScale) # + qZeroPt
    return weightInt.to(torch.float32)

def biasQuant(bias, qScale, qZeroPt):
    # The value "compensateScale" is defined in "prune.quantization"
    biasInt = torch.round(w_compensateScale * bias / qScale) + qZeroPt
    biasInt = biasInt.cpu().numpy()
    # If the weight is lower than lower_threshold, we set the weight to lower_threshold
    biasInt = np.where(biasInt <= 0, 0, biasInt)
    # If the weight is larger than upper_threshold, we set the weight to upper_threshold
    biasInt = np.where(biasInt >= 255, 255, biasInt)
    biasInt = torch.tensor(biasInt)
    return biasInt.to(torch.float32)

def batchFolding(conv, batchnorm):
    # batchnorm values

    bn_weight = batchnorm.weight # bn weights
    bn_mean = batchnorm.running_mean # u
    bn_var = batchnorm.running_var # sigma
    bn_eps = batchnorm.eps # eps
    bn_bias = batchnorm.bias # bias

    # conv values
    fused_conv = copy.deepcopy(conv)
    
    weight = conv.weight
    
    if conv.bias is None:
        bias = bn_mean.new_zeros(bn_mean.shape)
    else:
        bias = conv.bias

    # start folding
    bn_var_rsqrt = torch.rsqrt(bn_var + bn_eps) # rsqrt = 1 / sqrt
    bn_computation = (bn_weight * bn_var_rsqrt).view(-1, 1, 1, 1)
    Fweights = weight * bn_computation
    
    Fbias = (bias - bn_mean) * bn_var_rsqrt * bn_weight + bn_bias
    
    fused_conv.weight = torch.nn.Parameter(Fweights)
    fused_conv.bias = torch.nn.Parameter(Fbias)
    

    return fused_conv

def cfgReader(cfgFileName):
    dataList = []
    file = open(cfgFileName, "r")
    data = file.readlines()
    tmpList = []
    for i in range(len(data)):
        tmp = data[i].replace("\n", "").replace(" ", "").replace(")", "").replace("(", "").replace("]", "").replace("[", "").replace("tensor", "").split(",")
        tmpDict = tuple((float(tmp[0]), int(tmp[1]), float(tmp[2]), int(tmp[3])))
        tmpList.append(tmpDict)
    dataList.append(tmpList)
    file.close()
        
    return dataList[0]