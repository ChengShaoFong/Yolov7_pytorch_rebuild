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




##########################################################
########                  Basic                    #######
##########################################################

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        """在Focus、Bottleneck、BottleneckCSP、C3、SPP、DWConv、TransformerBloc等模組中呼叫
        Standard convolution  conv+BN+act
        :params c1: 輸入的channel值
        :params c2: 輸出的channel值
        :params k: 卷積的kernel_size
        :params s: 卷積的stride
        :params p: 卷積的padding  一般是None  可以透過autopad自行計算需要pad的padding數
        :params g: 卷積的groups數  =1就是普通的卷積  >1就是深度可分離卷積
        :params act: 啟用函式型別   True就是SiLU()/Swish   False就是不使用啟用函式
                     型別是nn.Module就使用傳進來的啟用函式型別
        """
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
##########################################################
##########           YOLOv7 Backbone            ##########
##########################################################

class ELAN_Block(nn.Module):
    """
    ELANBlock of YOLOv7 backbone
    """
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, depthwise=False, mask=True):
        super(ELAN_Block, self).__init__()
        
        inter_dim = int(in_dim * expand_ratio)
        self.conv1 = Conv(in_dim, inter_dim, kernel_size=1)   
        self.conv2 = Conv(in_dim, inter_dim, kernel_size=1)
        self.conv3 = nn.Sequential(
            Conv(inter_dim, inter_dim, kernel_size=3, padding=1, depthwise=depthwise),
            Conv(inter_dim, inter_dim, kernel_size=3, padding=1, depthwise=depthwise)
        )
        self.conv4 = nn.Sequential(
            Conv(inter_dim, inter_dim, kernel_size=3, padding=1, depthwise=depthwise),
            Conv(inter_dim, inter_dim, kernel_size=3, padding=1, depthwise=depthwise)
        )
        assert inter_dim * 4 == out_dim 
        self.out = Conv(inter_dim * 4, out_dim, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        #[B, C, H, W] => [B, 2C, H, W]
        out=self.out(torch.cat([x1, x2, x3, x4], dim=1)) #concat

        return out

class DownSample(nn.Module):
    def __init__(self, in_dim, mask=True):
        super().__init__()
        
        inter_dim = in_dim // 2
        self.mp = nn.MaxPool2d((2, 2), 2)
        self.conv1 = Conv(in_dim, inter_dim, kernel_size=1)
        self.conv2 = nn.Sequential(
            Conv(in_dim, inter_dim, kernel_size=1),
            Conv(inter_dim, inter_dim, kernel_size=3, padding=1, s=2)
        )

    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, C, H//2, W//2]
        """
        # [B, C, H, W] -> [B, C//2, H//2, W//2]
        x1 = self.conv1(self.mp(x))
        x2 = self.conv2(x)

        # [B, C, H//2, W//2]
        out = torch.cat([x1, x2], dim=1)

        return out
# ELANNet of YOLOv7
class ELANNet(nn.Module):
    """
    ELAN-Net of YOLOv7.
    """
    def __init__(self, depthwise=False, num_classes=1000, mask=True):
        super(ELANNet, self).__init__()
        
        self.layer_1 = nn.Sequential(
            Conv(3, 32, kernel_size=3, padding=1, depthwise=depthwise),      
            Conv(32, 64, kernel_size=3, padding=1, s=2, depthwise=depthwise),
            Conv(64, 64, kernel_size=3, padding=1, depthwise=depthwise)                                    # P1/2
        )
        self.layer_2 = nn.Sequential(   
            Conv(64, 128, kernel_size=3, padding=1, s=2, depthwise=depthwise),             
            ELAN_Block(in_dim=128, out_dim=256, expand_ratio=0.5, depthwise=depthwise)                     # P2/4
        )   
        self.layer_3 = nn.Sequential(
            DownSample(in_dim=256),             
            ELAN_Block(in_dim=256, out_dim=512, expand_ratio=0.5, depthwise=depthwise)                     # P3/8
        )
        self.layer_4 = nn.Sequential(
            DownSample(in_dim=512),             
            ELAN_Block(in_dim=512, out_dim=1024, expand_ratio=0.5, depthwise=depthwise)                    # P4/16
        )
        self.layer_5 = nn.Sequential(
            DownSample(in_dim=1024),             
            ELAN_Block(in_dim=1024, out_dim=1024, expand_ratio=0.25, depthwise=depthwise)                  # P5/32
        )

        self.avgpool = nn.AvgPool2d((1, 1))
        self.fc = nn.Linear(1024 , num_classes)  #50


    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)

        # [B, C, H, W] -> [B, C, 1, 1]
        x = self.avgpool(x)
        # [B, C, 1, 1] -> [B, C]
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

########################################################
###########           YOLOv7 Head            ###########
########################################################

class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)

        return self.cv7(torch.cat((y1, y2), dim=1))

class ELAN_HBlock(nn.Module):
    """
    ELANBlock of YOLOv7 backbone
    """
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, depthwise=False, act_type='silu', norm_type='BN',mask=True):
        super(ELAN_HBlock, self).__init__()
        
        inter_dim = int(in_dim * expand_ratio)
        inter_dim2 = int(inter_dim * expand_ratio)
        self.conv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.conv2 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.conv3 = Conv(inter_dim, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.conv4 = Conv(inter_dim2, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.conv5 = Conv(inter_dim2, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.conv6 = Conv(inter_dim2, inter_dim2, k=3, p=1, act_type=act_type, norm_type=norm_type, depthwise=depthwise)

        self.out = Conv(inter_dim*2 + inter_dim2*4, out_dim, kernel_size=1)

    def forward(self, x):
        """
        Input:
            x: [B, C_in, H, W]
        Output:
            out: [B, C_out, H, W]
        """
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

       # [B, C_in, H, W] -> [B, C_out, H, W]
        out = self.out(torch.cat([x1, x2, x3, x4, x5, x6], dim=1))

        return out

class DownSample_H(nn.Module):
    def __init__(self, in_dim, depthwise=False, act_type='silu', norm_type='BN',mask=True):
        super().__init__()
       
        inter_dim = in_dim
        self.mp = nn.MaxPool2d((2, 2), 2)
        self.conv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.conv2 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(inter_dim, inter_dim, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )

    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, 2C, H//2, W//2]
        """
        # [B, C, H, W] -> [B, C//2, H//2, W//2]
        x1 = self.conv1(self.mp(x))
        x2 = self.conv2(x)

        # [B, C, H//2, W//2]
        out = torch.cat([x1, x2], dim=1)

        return out

# PaFPN-ELAN (YOLOv7's)
class ELAN_HNet(nn.Module):
    def __init__(self, 
                 in_dims=[256, 512, 512],
                 out_dim=[256, 512, 1024],
                 depthwise=False,
                 norm_type='BN',
                 act_type='silu',
                 mask=True):
        super(ELAN_HNet, self).__init__()
        

        self.in_dims = in_dims
        self.out_dim = out_dim
        c3, c4, c5 = in_dims

        # top dwon
        ## P5 -> P4
        self.cv1 = Conv(c5, 256, k=1, norm_type=norm_type, act_type=act_type)
        self.cv2 = Conv(c4, 256, k=1, norm_type=norm_type, act_type=act_type)
        self.head_elan_1 = ELAN_HBlock(in_dim=256 + 256,
                                     out_dim=256,
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        # P4 -> P3
        self.cv3 = Conv(256, 128, k=1, norm_type=norm_type, act_type=act_type)
        self.cv4 = Conv(c3, 128, k=1, norm_type=norm_type, act_type=act_type)
        self.head_elan_2 = ELAN_HBlock(in_dim=128 + 128,
                                     out_dim=128,  # 128
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        # bottom up
        # P3 -> P4
        self.mp1 = DownSample(128, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.head_elan_3 = ELAN_HBlock(in_dim=256 + 256,
                                     out_dim=256,  # 256
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        # P4 -> P5
        self.mp2 = DownSample(256, act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.head_elan_4 = ELAN_HBlock(in_dim=512 + 512,
                                     out_dim=512,  # 512
                                     depthwise=depthwise,
                                     norm_type=norm_type,
                                     act_type=act_type)

        # RepConv
        self.repconv_1 = RepConv(128, out_dim[0], k=3, s=1, p=1)
        self.repconv_2 = RepConv(256, out_dim[1], k=3, s=1, p=1)
        self.repconv_3 = RepConv(512, out_dim[2], k=3, s=1, p=1)


    def forward(self, features):
        c3, c4, c5 = features

        # Top down
        ## P5 -> P4
        c6 = self.cv1(c5)
        c7 = F.interpolate(c6, scale_factor=2.0)  #upsample or downsample
        c8 = torch.cat([c7, self.cv2(c4)], dim=1)
        c9 = self.head_elan_1(c8)
        ## P4 -> P3
        c10 = self.cv3(c9)
        c11 = F.interpolate(c10, scale_factor=2.0)
        c12 = torch.cat([c11, self.cv4(c3)], dim=1)
        c13 = self.head_elan_2(c12)

        # Bottom up
        # p3 -> P4
        c14 = self.mp1(c13)
        c15 = torch.cat([c14, c9], dim=1)
        c16 = self.head_elan_3(c15)
        # P4 -> P5
        c17 = self.mp2(c16)
        c18 = torch.cat([c17, c5], dim=1)
        c19 = self.head_elan_4(c18)

        # RepCpnv
        c20 = self.repconv_1(c13)
        c21 = self.repconv_2(c16)
        c22 = self.repconv_3(c19)

        out_feats = [c20, c21, c22] # [P3, P4, P5]

        return out_feats

class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super(RepConv, self).__init__()

        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2

        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)

        else:
            self.rbr_identity = (nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None)

            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d( c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)  

class Detect(nn.Module):
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

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
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
            out = (z, )
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)

        return out