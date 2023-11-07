from modelConvertTool.getInfo import *
from modelConvertTool.modelModules import *
from utils.autoanchor import check_anchor_order

from pathlib import Path
from copy import deepcopy
from utils.general import make_divisible
import logging
import math

logger = logging.getLogger(__name__)

# the model builder
class modelBuilder(nn.Module):   # concat all class to run
    def __init__(self, model, modeltype, cfg=None, ch=3, nc=True, anchors=True, device ='cpu'):
        super(modelBuilder, self).__init__()

        
        myModel = getModel(model ,modeltype)

        self.mdList = []

        for item in myModel:
            self.mdList.append(item)
        
        # Define model
        self.model = nn.Sequential(OrderedDict(self.mdList)).to(device)
      
        m = self.model[-1]  # Detect()
        
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            # m.stride = torch.tensor([8., 16., 32.])

            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            # self._initialize_biases()  # only run once
            print('Strides: %s' % m.stride.tolist())      
        

    # PTQ不用init bias 
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):
        x = self.model(x)
        return x


# detect cvtlatest model owend layer module
def getModel(model ,modeltype):  

    if modeltype == "yolov7":
        myModel0 = []    # for ELANNet
        myModel1 = []    # for ELANHNet
        myModel = []
        convMod = None
        LayerIdx = 0
        getMP = False
        getELAN = False
     
        for name, module in model.named_children():  # Model
            moduleStr = str(module)
            if "Sequential" in moduleStr:
                for L0name, L0mod in module.named_children(): 
                    # print('totalLayer',totalLayer)
                    # totalLayer += 1
                    L0modStr = str(L0mod)

                    if  L0modStr.startswith("Conv(") and LayerIdx < 4:

                        myModel0.append(("Conv"+str(LayerIdx), L0mod))
                        LayerIdx += 1
                        continue
                    ########################################### ELAN ###################################################
                    if  L0modStr.startswith("Conv(") and LayerIdx in (4, 17, 30, 43):

                        storeELAN = []
                        storeELAN.append(L0mod) #0
                        getELAN = True
                        ELANcount = 0
                        LayerIdx += 1 
                        continue
                
                    if  L0modStr.startswith("Conv(") and getELAN and LayerIdx <= 50:
                        
                        if ELANcount >= 5 :   # 5 because have one concat
                
                            storeELAN.append(L0mod) #6
                            myModel0.append(("ELAN_block"+str(LayerIdx-7)+"~"+str(LayerIdx), storeELAN))
                            getELAN = False
                            LayerIdx += 1
                        
                        else :

                            storeELAN.append(L0mod) # 1, 2, 3, 4, 5
                            ELANcount += 1  
                            LayerIdx += 1

                        continue
                    ######################################### MP1 (DownSample) #########################################
                    if  L0modStr.startswith("MP(") and LayerIdx <= 50:
                        storeMP = []
                        storeMP.append(L0mod) #0
                        getMP= True
                        MPcount = 0
                        LayerIdx += 1
                        continue

                    if  (L0modStr.startswith("Conv(") or L0modStr.startswith("Concat(")) and getMP and LayerIdx <= 50 :

                        if MPcount >= 3:
                            
                            myModel0.append(("maxpool"+str(LayerIdx-4)+"~"+str(LayerIdx), storeMP))
                            getMP=False
                            LayerIdx += 1
                            
                        else :
                            
                            storeMP.append(L0mod) # 1, 2, 3,
                            MPcount += 1  
                            LayerIdx += 1

                        continue

                    if  LayerIdx == 51  :
                        
                        ELANNetmod = getELANNet(myModel0) 
                       
                        myModel.append(("ELANNet", ELANNetmod))

                    ############################################### SPPCSPC ############################################
                    if L0modStr.startswith("SPPCSPC("):
    
                        myModel1.append(("SPPCSPC"+str(LayerIdx), L0mod))  # change module append 
                        LayerIdx += 1
                        continue
                    
                    ############################################ Upsample ##############################################
                    if L0modStr.startswith("Conv(") and LayerIdx in (52, 54, 64, 66):

                        myModel1.append(("Conv"+str(LayerIdx), L0mod))
                        LayerIdx += 1
                        continue
                    
                    if L0modStr.startswith("Upsample("):

                        myModel1.append(("upsample"+str(LayerIdx), L0mod))
                        LayerIdx += 1
                        continue
                    ############################################ ELAN_H ##############################################
                    if L0modStr.startswith("Conv(") and LayerIdx in (56, 68, 81, 94):

                        storeELAN_H = []
                        storeELAN_H.append(L0mod) #0
                        getELAN = True
                        ELAN_Hcount = 0
                        LayerIdx += 1 
                        continue

                    if L0modStr.startswith("Conv(") and getELAN and LayerIdx > 50:
                        
                        if ELAN_Hcount >= 5 :   # 5 because have one concat
                            
                            storeELAN_H.append(L0mod) #6
                            myModel1.append(("ELAN_Hblock"+str(LayerIdx-7)+"~"+str(LayerIdx), storeELAN_H))
                            getELAN = False
                            LayerIdx += 1
                        
                        else :

                            storeELAN_H.append(L0mod) # 1, 2, 3, 4, 5
                            ELAN_Hcount += 1  
                            LayerIdx += 1

                        continue
                    
                    ############################################ MP2 (Downsample) #######################################
                    if L0modStr.startswith("MP(") and LayerIdx > 50:
                        storeMP2 = []
                        storeMP2.append(L0mod) #0
                        getMP= True
                        MPcount = 0
                        LayerIdx += 1
                        continue

                    if (L0modStr.startswith("Conv(") or L0modStr.startswith("Concat(")) and getMP and LayerIdx > 50 :

                        if MPcount >= 3:
                            
                            myModel1.append(("maxpool"+str(LayerIdx-4)+"~"+str(LayerIdx), storeMP2))
                            getMP=False
                            LayerIdx += 1
                            
                        else :

                            storeMP2.append(L0mod) # 1, 2, 3,
                            MPcount += 1  
                            LayerIdx += 1
                        continue

                    ############################################ RepConv ################################################
                    if L0modStr.startswith("RepConv("):

                        myModel1.append(("RepConv"+str(LayerIdx), L0mod))
                        LayerIdx += 1

                        continue
                        
                    ########################################### ELAN_HNet#############################################
                    if LayerIdx == 105 :
                
                        ELAN_HNetmod = getELAN_HNet(myModel1)  
                        myModel.append(("ELAN_HNet", ELAN_HNetmod))

                    ############################################ Detect ###############################################
                    if L0modStr.startswith("Detect("):
                        Detectmod = getDetectModule(L0mod)
                       
                        myModel.append(("Detect"+str(LayerIdx), Detectmod))
                        LayerIdx += 1          
                        continue
            
                    else:   # concat

                        LayerIdx += 1
                        continue
    
    # resModel = modelBuilder(myModel, modeltype=modeltype)

    return myModel

# build convolution layer module - resnet
def getConvModule(convMod, bnMod):
    # Get convolution information
    inChannel, outChannel, kernel_size, stride, padding, weight, bias = getConvInfo(convMod)
    biasBool = getBiasBool(bias)
    # Get batch-normalization information
    bn_weight, bn_mean, bn_var, bn_eps, bn_bias = getBNInfo(bnMod)
    
    # Build up convolution module
    convModule = convLayer(inChannel, outChannel, kernel_size, stride, padding, biasBool)
    # Fill the following weight and bias
    for name, module in convModule.named_children():
        if "conv" in name:
            # for conv module
            module.conv.weight = weight
            module.conv.bias = bias
            
            # for batchnorm module
            module.batchnorm.weight = bn_weight
            module.batchnorm.bias = bn_bias
            module.batchnorm.running_mean = bn_mean
            module.batchnorm.running_var = bn_var
            module.batchnorm.eps = bn_eps
            
    return convModule

# build fully-connected layer module - resnet
def getFcModule(fcMod, isReLU=False):
    # Get fully-connected information
    inChannel, outChannel, weight, bias = getFcInfo(fcMod)
    # Build up fully-connected module
    fcModule = fcLayer(inChannel, outChannel, isReLU)
    # fcModule = nn.Linear(inChannel, outChannel)
    # Fill the following weight and bias
    fcModule.fc.weight = weight
    fcModule.fc.bias = bias
    return fcModule

####################### YOLOV7 #############################

def getELANNet(ELANNetmod): # total 81 layer conv+bn

    ELANNetInfo = getELANNetInfo(ELANNetmod)
   
    basicInfo = ELANNetInfo[69]
 
    ElANNetblock = ELANNet()
    idx = 0

    for name, module in ElANNetblock.named_children():   #check ELANNet
  
        if name.startswith("cv") :   # follow ELANNet     
            for name0, module0 in module.named_children():
            
                if "conv" in name0:
                    
                # for conv module
                    module0.weight = ELANNetInfo[idx][1][5]
                    module0.bias = ELANNetInfo[idx][1][6]
                    idx += 1

                if "bn" in name0:
         
                # for batchnorm module
                    module0.weight = ELANNetInfo[idx][1][0]
                    module0.bias = ELANNetInfo[idx][1][4]
                    module0.running_mean = ELANNetInfo[idx][1][1]
                    module0.running_var = ELANNetInfo[idx][1][2]
                    module0.eps = ELANNetInfo[idx][1][3]
                    idx += 1
                    continue

                if "act" in name0:
                    module0 = nn.SiLU(inplace=True)

                    continue

        if name.startswith("elan") or name.startswith("down"):
            
            for name0, module0 in module.named_children():  
                module0Str = str(module0)

                if module0Str.startswith("Conv("):
                    # print("into ELAN_conv")
                    for name1, module1 in module0.named_children(): 
                   
                        if "conv" in name1:
                        # for conv module
                            module1.weight = ELANNetInfo[idx][1][5]
                            module1.bias = ELANNetInfo[idx][1][6]
                            idx += 1

                        if "bn" in name1:
                        # for batchnorm module
                            
                            module1.weight = ELANNetInfo[idx][1][0]
                            module1.bias = ELANNetInfo[idx][1][4]
                            module1.running_mean = ELANNetInfo[idx][1][1]
                            module1.running_var = ELANNetInfo[idx][1][2]
                            module1.eps = ELANNetInfo[idx][1][3]
                            idx += 1
                            continue

                        if "act" in name1:
                            module1 = nn.SiLU(inplace=True)
                            continue

    return ElANNetblock   # got 1~50 layer and have 3 output              
               
def getELAN_HNet(ELAN_HNetmod):  # total 101 layer conv+bn
    ELAN_HNetInfo = getELAN_HNetInfo(ELAN_HNetmod)
    
    ElAN_HNetblock = ELAN_HNet()
    # print(ELAN_HNetInfo[102]) 
    idx = 0
    for name, module in ElAN_HNetblock.named_children():
       
        if name.startswith("cv"):
            for name0, module0 in module.named_children():
              
                if "conv" in name0:
                # for conv module
                    module0.weight = ELAN_HNetInfo[idx][1][5]
                    module0.bias = ELAN_HNetInfo[idx][1][6]
                    idx += 1

                if "bn" in name0:
                # for batchnorm module
                    module0.weight = ELAN_HNetInfo[idx][1][0]
                    module0.bias = ELAN_HNetInfo[idx][1][4]
                    module0.running_mean = ELAN_HNetInfo[idx][1][1]
                    module0.running_var = ELAN_HNetInfo[idx][1][2]
                    module0.eps = ELAN_HNetInfo[idx][1][3]
                    idx += 1
                    continue

                if "act" in name0:
                    module0 = nn.SiLU(inplace=True)
                    continue

        if name.startswith("head_elan") or name.startswith("down") or name.startswith("SPPCSPC"):
            for name0, module0 in module.named_children():
                module0Str = str(module0)

                if "Conv" in module0Str:
                    for name1, module1 in module0.named_children():

                        if "conv" in name1:
                        # for conv module
                            module1.weight = ELAN_HNetInfo[idx][1][5]
                            module1.bias = ELAN_HNetInfo[idx][1][6]
                            idx += 1

                        if "bn" in name1:
                        # for batchnorm module
                            module1.weight = ELAN_HNetInfo[idx][1][0]
                            module1.bias = ELAN_HNetInfo[idx][1][4]
                            module1.running_mean = ELAN_HNetInfo[idx][1][1]
                            module1.running_var = ELAN_HNetInfo[idx][1][2]
                            module1.eps = ELAN_HNetInfo[idx][1][3]
                            idx += 1
                            continue

                        if "act" in name1:
                            module1 = nn.SiLU(inplace=True)
                            continue

        if name.startswith("repconv"):
            for name0, module0 in module.named_children():
             
                module0Str = str(module0)
                
                if "Sequential" in module0Str :
                    for name1, module1 in module0.named_children():
                        module1Str = str(module1)

                        if "Conv" in module1Str :
                        
                        # for conv module
                            module1.weight = ELAN_HNetInfo[idx][1][5]
                            module1.bias = ELAN_HNetInfo[idx][1][6]
                            idx += 1

                        if "BatchNorm" in module1Str :
                            
                        # for batchnorm module
                            module1.weight = ELAN_HNetInfo[idx][1][0]
                            module1.bias = ELAN_HNetInfo[idx][1][4]
                            module1.running_mean = ELAN_HNetInfo[idx][1][1]
                            module1.running_var = ELAN_HNetInfo[idx][1][2]
                            module1.eps = ELAN_HNetInfo[idx][1][3]
                            
                            idx += 1
                            continue

                if "act" in name0:
                    module0 = nn.SiLU(inplace=True)
                    continue
        
        
    return ElAN_HNetblock
    # for name, module in ElAN_HNetblock.named_children():   #check ELANNet

def getDetectModule(Detectmod):
    DetectInfo = getDectectInfo(Detectmod)   # the convblock is mean conv + bn + silu is group
  
    basicInfo = [DetectInfo[0][1][0], DetectInfo[1][1][0] , DetectInfo[2][1][0]]

    #in_channels, out_channels, kernel_size, stride, padding, weight ,bias 
    
    anchors = torch.tensor([[12, 16, 19, 36, 40, 28], [36, 75, 76, 55, 72, 146], [142, 110, 192, 243, 459, 401]])
    Detblcok = Detect(anchors=anchors, ch=basicInfo, nc=80)  #ch = 256, 512, 1024
    idx = 0
    # print(DetectInfo[3])
    for name, module in Detblcok.named_children():
        moduleStr = str(module)
        
        if "ModuleList" in moduleStr : 
            for L0name , L0module in module.named_children():
                L0moduleStr = str(L0module)

                if "Conv2d" in L0moduleStr:

                    L0module.weight = DetectInfo[idx][1][5]
                    L0module.bias = DetectInfo[idx][1][6]   

                    idx += 1
                    continue
    
    return Detblcok

####################### yaml ##############################
def parse_model(d, ch):  # model_dict, input_channels(3)
        logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
        anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
        na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
        no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

        layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
        for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
       
            m = eval(m) if isinstance(m, str) else m  # eval strings
            for j, a in enumerate(args):
                try:
                
                    args[j] = eval(a) if isinstance(a, str) else a  # eval strings
                except:
                    pass
           
            n = max(round(n * gd), 1) if n > 1 else n  # depth gain

            
            if m in [nn.Conv2d, Conv, SPPCSPC, RepConv]:
                c1, c2 = ch[f], args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)

                args = [c1, c2, *args[1:]]
                if m in [SPPCSPC]:
                    args.insert(2, n)  # number of repeats
                    n = 1   
                    
            elif m is nn.BatchNorm2d:
                args = [ch[f]]

            elif m is Concat:
                c2 = sum([ch[x] for x in f])

            elif m in [Detect]:
                args.append([ch[x] for x in f])
                if isinstance(args[1], int):  # number of anchors
                    args[1] = [list(range(args[1] * 2))] * len(f) 
            else:
                c2 = ch[f]

            m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
            t = str(m)[8:-2].replace('__main__.', '')  # module type
            np = sum([x.numel() for x in m_.parameters()])  # number params
            m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
            logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)
        
        return nn.Sequential(*layers), sorted(save)


