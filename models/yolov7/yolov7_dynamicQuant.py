import torch.nn as nn
from collections import OrderedDict
from prune.pruning import *
import warnings 
from prune.pruning_uint8 import *
from prune.quantization import *

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class v7convLayer(PruningModule):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, isOffsetReset=True): # ch_in, ch_out, kernel, stride, padding, groups
        super(v7convLayer, self).__init__()
        self.isOffsetReset = isOffsetReset
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2 , eps=0.001, momentum=0.03)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, PruningModule) else nn.Identity())

    def forward(self, x : Tensor, accumScale = 1.0, isLayer1=False) -> Tensor :
        tmpList = []
        
        x, weight, xScale, xZeroPt, convScale = inputQuantizer(x, self.conv, accumScale, isLayer1=isLayer1, types="conv")
        x = self.conv(x)
        x = self.bn(x)
        x = floatConverter(x, xScale, self.conv)

        x = self.act(x)
        x, accumScale, accumZeroPt = accumQuantizer(x, isOffsetReset=self.isOffsetReset)
        tmpList.append((float(xScale), int(xZeroPt), float(accumScale), int(accumZeroPt), float(convScale)))

        return x ,accumScale, accumZeroPt, tmpList

        # return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class DownSample(PruningModule):
    def __init__(self, in_dim, mask=True):
        super().__init__()
        
        inter_dim = in_dim // 2
        self.mp = nn.MaxPool2d((2, 2), 2)
        self.cv1 = v7convLayer(in_dim, inter_dim, k=1)

        self.cv2 = v7convLayer(in_dim, inter_dim, k=1)
        self.cv3 = v7convLayer(inter_dim, inter_dim, k=3, p=1, s=2)

    def forward(self, x):
        blockList = []
        ## 
        a = self.mp(x)

        x1, accumScale, accumZeroPt, tmpList = self.cv1(a, accumScale)
        blockList.append(tmpList[0])

        ## 
        b, accumScale, accumZeroPt, tmpList = self.cv2(x, accumScale)
        blockList.append(tmpList[0])

        x2, accumScale, accumZeroPt, tmpList = self.cv3(b, accumScale)
        blockList.append(tmpList[0])

        ##
        out = torch.cat([x2, x1], dim=1)

        return out, accumScale, accumZeroPt, blockList
    
class DownSample_H(PruningModule):
    def __init__(self, in_dim, mask=True):
        super().__init__()
        
        inter_dim = in_dim
        self.mp = nn.MaxPool2d((2, 2), 2)
        self.cv1 = v7convLayer(in_dim, inter_dim, k=1)

        self.cv2 = v7convLayer(in_dim, inter_dim, k=1)
        self.cv3 = v7convLayer(inter_dim, inter_dim, k=3, p=1, s=2)
        

    def forward(self, x):

        x1 = self.cv1(self.mp(x))
        x2 = self.cv3(self.cv2(x))

        out = torch.cat([x2, x1], dim=1)

        return out

class SPPCSPC(PruningModule):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = v7convLayer(c1, c_, 1, 1)
        self.cv2 = v7convLayer(c1, c_, 1, 1)
        self.cv3 = v7convLayer(c_, c_, 3, 1)
        self.cv4 = v7convLayer(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = v7convLayer(4 * c_, c_, 1, 1)
        self.cv6 = v7convLayer(c_, c_, 3, 1)
        self.cv7 = v7convLayer(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
     
        return self.cv7(torch.cat((y1, y2), dim=1))

class RepLayer(PruningModule):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super(RepLayer, self).__init__()

        self.deploy = deploy
        self.groups = g
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(k, int):
            padding_11 = autopad(k, p) - k // 2
        else:
            padding_11 = tuple([x - y // 2 for x, y in zip(autopad(k, p), k)])

        self.act = nn.SiLU() if act is True else (act if isinstance(act, PruningModule) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels, out_channels, k, s, autopad(k, p), groups=g, bias=True)
        else:
            self.rbr_identity = (nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and s == 1 else None)
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=out_channels, momentum=0.03),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d( in_channels, out_channels, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=out_channels, momentum=0.03),
            )

    def forward(self, x):
        if hasattr(self, "rbr_reparam"):
            x = self.act(x)
            x = self.rbr_reparam(x)
            return x

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(x)
       
        return self.act(self.rbr_dense(x) + self.rbr_1x1(x) + id_out)  

#ELAN_Block of YOLOv7
class ELAN_Block(PruningModule):
    """
    ELANBlock of YOLOv7 backbone
    """
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, depthwise=False, mask=True):
        super(ELAN_Block, self).__init__()
    
        inter_dim = int(in_dim * expand_ratio)
        
        self.conv1 = v7convLayer(in_dim, inter_dim, k=1)   
        self.conv2 = v7convLayer(in_dim, inter_dim, k=1)
        
        # blcok 1
        self.conv3 = v7convLayer(inter_dim, inter_dim, k=3, p=1)
        self.conv4 = v7convLayer(inter_dim, inter_dim, k=3, p=1)

        # block 2
        self.conv5 = v7convLayer(inter_dim, inter_dim, k=3, p=1)
        self.conv6 = v7convLayer(inter_dim, inter_dim, k=3, p=1)
        
        # assert inter_dim * 4 == out_dim 
        self.out = v7convLayer(inter_dim * 4, out_dim  , k=1)
        

    def forward(self, x):  # batchsize, in_dim, height, width 
        blockList = []
        x1, accumScale, accumZeroPt ,tmpList = self.conv1(x, accumScale)
        blockList.append(tmpList[0])

        x2, accumScale, accumZeroPt ,tmpList = self.conv2(x, accumScale)
        blockList.append(tmpList[0])

        x3, accumScale, accumZeroPt ,tmpList = self.conv3(x2, accumScale)
        blockList.append(tmpList[0])

        x4, accumScale, accumZeroPt ,tmpList = self.conv4(x3, accumScale)
        blockList.append(tmpList[0])

        x5, accumScale, accumZeroPt ,tmpList = self.conv5(x4, accumScale)
        blockList.append(tmpList[0])

        x6, accumScale, accumZeroPt ,tmpList = self.conv6(x5, accumScale)
        blockList.append(tmpList[0])
  
        out =torch.cat([x6, x4, x2, x1], dim=1)

        out, accumScale, accumZeroPt ,tmpList = self.out(out, accumScale)
        blockList.append(tmpList[0])

        return out, accumScale, accumZeroPt, blockList

# ELANNet of YOLOv7
class ELANNet(PruningModule):
    """
    ELAN-Net of YOLOv7.
    """
    def __init__(self, depthwise=False, num_classes=1000, mask=True):
        super(ELANNet, self).__init__()
        
        self.cv1 = v7convLayer(3, 32, k=3, p=1)     
        self.cv2 = v7convLayer(32, 64, k=3, p=1, s=2)
        self.cv3 = v7convLayer(64, 64, k=3, p=1)                   # P1/2
        self.cv4 = v7convLayer(64, 128, k=3, p=1, s=2)
        self.elan1 = ELAN_Block(128, 256, expand_ratio=0.5)           # P2/4

        self.down1 = DownSample(in_dim=256)
        self.elan2 = ELAN_Block(in_dim=256, out_dim=512, expand_ratio=0.5)
       
        self.down2 = DownSample(in_dim=512)           
        self.elan3 = ELAN_Block(in_dim=512, out_dim=1024, expand_ratio=0.5)                    # P4/16
        
        self.down3 = DownSample(in_dim=1024)          
        self.elan4 = ELAN_Block(in_dim=1024, out_dim=1024, expand_ratio=0.25)                  # P5/32
        
        
        # self.avgpool = nn.AvgPool2d((1, 1))
        # self.fc = nn.Linear(1024 * 10 * 21 , num_classes)  #50

    def forward(self, x:Tensor, idx) -> Tensor:
      
        quantList = []
        #cv1
        x, accumScale, accumZeroPt ,tmpList = self.cv1(x, accumScale=1.0, isLayer1=True)
        quantList.append(tmpList[0])

        x, accumScale, accumZeroPt ,tmpList = self.cv2(x, accumScale)
        quantList.append(tmpList[0])

        x, accumScale, accumZeroPt ,tmpList = self.cv3(x, accumScale)
        quantList.append(tmpList[0])

        x, accumScale, accumZeroPt ,tmpList = self.cv4(x, accumScale)
        quantList.append(tmpList[0])

        x1, accumScale, accumZeroPt ,tmpList = self.elan1(x, accumScale)
        quantList.append(tmpList[0])

        # out 1
        x2, accumScale, accumZeroPt ,tmpList = self.down1(x1, accumScale)
        quantList.append(tmpList[0])

        x2, accumScale, accumZeroPt ,tmpList = self.elan2(x2, accumScale)
        quantList.append(tmpList[0])
        
        # out 2
        x3, accumScale, accumZeroPt ,tmpList = self.down2(x2, accumScale)
        quantList.append(tmpList[0])

        x3, accumScale, accumZeroPt ,tmpList = self.elan3(x3, accumScale)
        quantList.append(tmpList[0])
        
        # out 3
        x4, accumScale, accumZeroPt ,tmpList = self.down3(x3, accumScale)
        quantList.append(tmpList[0])

        x4, accumScale, accumZeroPt ,tmpList = self.elan4(x4, accumScale)
        quantList.append(tmpList[0])


        feature = [x2, x3, x4]

        return feature ,accumScale, accumZeroPt, quantList

#ELAN_HBlock of YOLOv7
class ELAN_HBlock(PruningModule):
    """
    ELANBlock of YOLOv7 backbone
    """
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, depthwise=False, act_type='silu', norm_type='BN',mask=True):
        super(ELAN_HBlock, self).__init__()
        
        inter_dim = int(in_dim * expand_ratio)
        inter_dim2 = int(inter_dim * expand_ratio)
        self.conv1 = v7convLayer(in_dim, inter_dim, k=1)
        self.conv2 = v7convLayer(in_dim, inter_dim, k=1)
        self.conv3 = v7convLayer(inter_dim, inter_dim2, k=3, p=1)
        self.conv4 = v7convLayer(inter_dim2, inter_dim2, k=3, p=1)
        self.conv5 = v7convLayer(inter_dim2, inter_dim2, k=3, p=1)
        self.conv6 = v7convLayer(inter_dim2, inter_dim2, k=3, p=1)

        self.out = v7convLayer(inter_dim*2 + inter_dim2*4, out_dim, k=1)

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
      
       # [B, C_in, H, W] -> [B, C_out, H, W]
        out = self.out(torch.cat([x6, x5, x4, x3, x2, x1], dim=1))
        
        return out

# ELAN_HNet of YOLOv7
class ELAN_HNet(PruningModule):
    def __init__(self, in_dims=[512, 1024, 1024], out_dim=[256, 512, 1024], depthwise=False, mask=True):     #  norm_type='BN', act_type='silu',
        super(ELAN_HNet, self).__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        c3, c4, c5 = in_dims
       
        # top dwon
        ## P5 -> P4
        self.SPPCSPC = SPPCSPC(c5, 512, k=[5, 9, 13])

        # Upsample
        self.cv1 = v7convLayer(512, 256, k=1)
        self.cv2 = v7convLayer(c4, 256, k=1)
        self.head_elan_1 = ELAN_HBlock(in_dim=256 + 256,
                                       out_dim=256,
                                       depthwise=depthwise)
        # P4 -> P3
        self.cv3 = v7convLayer(256, 128, k=1)
        self.cv4 = v7convLayer(c3, 128, k=1)
        self.head_elan_2 = ELAN_HBlock(in_dim=128 + 128,
                                       out_dim=128,        # 128
                                       depthwise=depthwise)
        # bottom up
        # P3 -> P4
        self.down1 = DownSample_H(128)
        self.head_elan_3 = ELAN_HBlock(in_dim=256 + 256,
                                       out_dim=256,        # 256
                                       depthwise=depthwise)
        # P4 -> P5
        self.down2 = DownSample_H(256)
        self.head_elan_4 = ELAN_HBlock(in_dim=512 + 512,
                                       out_dim=512,        # 512
                                       depthwise=depthwise)
        # RepConv
        self.repconv_1 = RepLayer(128, out_dim[0], k=3, s=1, p=1)
        self.repconv_2 = RepLayer(256, out_dim[1], k=3, s=1, p=1)
        self.repconv_3 = RepLayer(512, out_dim[2], k=3, s=1, p=1)

    def forward(self, features):

        cfgFilePath = "./cfg/" + self.version + "/yolov7/" + self.dataset + "/" + str(w_quantBitwidth) + "-bit/" +  self.dataset + "_yolov7_trainObserver" + str(idx) + ".cfg"
        cfgFile = open(cfgFilePath, "w+")
        c3, c4, c5 = features[0]
        quantList = features[1]

        # Top down
        ## P5 -> P4
        cSP = self.SPPCSPC(c5)
        
        c6 = self.cv1(cSP)  #1024 -> 512
        c7 = F.interpolate(c6, scale_factor=2.0)  #upsample or downsample  #512 -> 512
        c8 = torch.cat([self.cv2(c4), c7], dim=1)
        c9 = self.head_elan_1(c8)

        ## P4 -> P3    (c10, c11)=upsample
        c10 = self.cv3(c9)
        c11 = F.interpolate(c10, scale_factor=2.0) # tensor sclae double the size
        c12 = torch.cat([self.cv4(c3), c11], dim=1)
        c13 = self.head_elan_2(c12)

        # Bottom up
        # p3 -> P4
        c14 = self.down1(c13)
        c15 = torch.cat([c14, c9], dim=1) # 256 -> 512
        c16 = self.head_elan_3(c15)

        # P4 -> P5
        c17 = self.down2(c16)
        c18 = torch.cat([c17, cSP], dim=1)
        c19 = self.head_elan_4(c18)

        # RepCpnv
        c20 = self.repconv_1(c13)
        c21 = self.repconv_2(c16)
        c22 = self.repconv_3(c19)

        out_feats = [c20, c21, c22] # [P3, P4, P5]

        return out_feats

# Detect
class Detect(PruningModule):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False
    
    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        
    def forward(self, x):   # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        
        for i in range(self.nl):
            
            
            x[i] = self.m[i](x[i])  # conv  #圖像特徵提取 self.m[i] 為模型卷積層 x[i] 是輸入到該層的特徵圖
            
            #該行程式碼的作用是將 x[i] 作為輸入傳遞給卷積層 self.m[i] 中進行卷積運算，並將結果再次賦值給 x[i]，從而實現特徵圖的更新。這是在進行前向傳播運算時進行的。

            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)    
            
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
       
            if not self.training:  # inference
              
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                   
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
              
                if not torch.onnx.is_in_onnx_export():  

                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh     
                else:
               
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                    y = torch.cat((xy, wh, conf), 4)

                z.append(y.view(bs, -1, self.no))
            
        if self.training:
            out = x
           
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, x)
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)
            
        return out

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=z.device)
        box @= convert_matrix                          
        return (box, score)



# Quantize Function

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
