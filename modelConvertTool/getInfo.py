# get convolution information
def getConvInfo(conv):
    inChannel = conv.in_channels
    outChannel = conv.out_channels
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    weight = conv.weight
    
    bias = conv.bias
    return inChannel, outChannel, kernel_size, stride, padding, weight, bias

# get batch normalization information
def getBNInfo(batchnorm):
    bn_weight = batchnorm.weight # bn weights
    bn_mean = batchnorm.running_mean # mu
    bn_var = batchnorm.running_var # sigma
    bn_eps = batchnorm.eps # eps
    bn_bias = batchnorm.bias # bias
    return bn_weight, bn_mean, bn_var, bn_eps, bn_bias

# get fully-connected information
def getFcInfo(module):
    inFeature = module.in_features
    outFeature = module.out_features
    weight = module.weight
    bias = module.bias
    
    return inFeature, outFeature, weight, bias

# def getMaxPoolInfo(module):
#     kernel_size = module.kernel_size
#     stride = module.stride
#     padding = module.padding
#     dilation = module.dilation
#     ceil_mode = False
#     return kernel_size, stride, padding, dilation, ceil_mode

# check bias or not
def getBiasBool(bias):
    if bias is None:
        biasBool = False
    else:
        biasBool = True
    return biasBool

# get Resblock's information
def getResBlockInfo(resblock):
    resInfo = []
    downSampleInfo = []
    for name, module in resblock.named_children():
        if "conv" in name:
            inChannel, outChannel, kernel_size, stride, padding, weight, bias = getConvInfo(module)
            resInfo.append((name, (inChannel, outChannel, kernel_size, stride, padding, weight, bias)))
            continue
        if "bn" in name:
            bn_weight, bn_mean, bn_var, bn_eps, bn_bias = getBNInfo(module)
            resInfo.append((name, (bn_weight, bn_mean, bn_var, bn_eps, bn_bias)))
            continue
        if "downsample" in name:
            inChannel, outChannel, kernel_size, stride, padding, weight, bias = getConvInfo(module[0])
            downSampleInfo.append(("conv", (inChannel, outChannel, kernel_size, stride, padding, weight, bias)))
            bn_weight, bn_mean, bn_var, bn_eps, bn_bias = getBNInfo(module[1])
            downSampleInfo.append(("batchnorm", (bn_weight, bn_mean, bn_var, bn_eps, bn_bias)))
            continue
    return resInfo, downSampleInfo

# get yolov7's information
def getELANNetInfo(ELANNetmod):
    
    ELANNetInfo = []

    for i in ELANNetmod:
       
        if "Conv" in i[0]: 
           
            for name, module in i[1].named_children():
                
                if "conv" in name:
                    inChannel, outChannel, kernel_size, stride, padding, weight, bias = getConvInfo(module)
                    ELANNetInfo.append((name, (inChannel, outChannel, kernel_size, stride, padding, weight, bias)))
                    continue

                if "bn" in name:
                    bn_weight, bn_mean, bn_var, bn_eps, bn_bias = getBNInfo(module)
                    ELANNetInfo.append((name, (bn_weight, bn_mean, bn_var, bn_eps, bn_bias)))
                    continue

        if "ELAN_block" in i[0]:           
            
            for j in i[1]:
    
                for name, module in j.named_children():
                    
                    if "conv" in name:
                        inChannel, outChannel, kernel_size, stride, padding, weight, bias = getConvInfo(module)
                        ELANNetInfo.append((name, (inChannel, outChannel, kernel_size, stride, padding, weight, bias)))
                        continue

                    if "bn" in name:
                        bn_weight, bn_mean, bn_var, bn_eps, bn_bias = getBNInfo(module)
                        ELANNetInfo.append((name, (bn_weight, bn_mean, bn_var, bn_eps, bn_bias)))
                        continue

        if "maxpool" in i[0]:

            for j in i[1]:
       
                for name, module in j.named_children():
              
                    if "conv" in name:
                        inChannel, outChannel, kernel_size, stride, padding, weight, bias = getConvInfo(module)
                        ELANNetInfo.append((name, (inChannel, outChannel, kernel_size, stride, padding, weight, bias)))
                        continue
                    
                    if "bn" in name:
                        bn_weight, bn_mean, bn_var, bn_eps, bn_bias = getBNInfo(module)
                        ELANNetInfo.append((name, (bn_weight, bn_mean, bn_var, bn_eps, bn_bias)))
                        continue

    return ELANNetInfo

def getELAN_HNetInfo(ELAN_HNetmod):
    ELAN_HNetInfo = []

    for i in ELAN_HNetmod :
        
        if "Conv" in i[0]:

            for name, module in i[1].named_children():
                
                if "conv" in name:
                    inChannel, outChannel, kernel_size, stride, padding, weight, bias = getConvInfo(module)
                    ELAN_HNetInfo.append((name, (inChannel, outChannel, kernel_size, stride, padding, weight, bias)))
                    continue
                if "bn" in name:
                    bn_weight, bn_mean, bn_var, bn_eps, bn_bias = getBNInfo(module)
                    ELAN_HNetInfo.append((name, (bn_weight, bn_mean, bn_var, bn_eps, bn_bias)))
                    continue

        if "SPPCSPC" in i[0]:

            for name, module in i[1].named_children():
                moduleStr = str(module)
                if "Conv" in moduleStr:
                    for name0, module0 in module.named_children():
                        
                        if "conv" in name0:
                            inChannel, outChannel, kernel_size, stride, padding, weight, bias = getConvInfo(module0)
                            ELAN_HNetInfo.append((name0, (inChannel, outChannel, kernel_size, stride, padding, weight, bias)))
                            continue

                        if "bn" in name0:
                            bn_weight, bn_mean, bn_var, bn_eps, bn_bias = getBNInfo(module0)
                            ELAN_HNetInfo.append((name0, (bn_weight, bn_mean, bn_var, bn_eps, bn_bias)))
                            continue

        if "ELAN_Hblock" in i[0]:

            for j in i[1]:
              
                for name, module in j.named_children():
                    
                    if "conv" in name:
                        inChannel, outChannel, kernel_size, stride, padding, weight, bias = getConvInfo(module)
                        ELAN_HNetInfo.append((name, (inChannel, outChannel, kernel_size, stride, padding, weight, bias)))
                        continue
                    if "bn" in name:
                        bn_weight, bn_mean, bn_var, bn_eps, bn_bias = getBNInfo(module)
                        ELAN_HNetInfo.append((name, (bn_weight, bn_mean, bn_var, bn_eps, bn_bias)))
                        continue

        if "maxpool" in i[0]:
            
            for j in i[1]:
                for name, module in j.named_children():
                    if "conv" in name:
                        inChannel, outChannel, kernel_size, stride, padding, weight, bias = getConvInfo(module)
                        ELAN_HNetInfo.append((name, (inChannel, outChannel, kernel_size, stride, padding, weight, bias)))
                        continue
                    if "bn" in name:
                        bn_weight, bn_mean, bn_var, bn_eps, bn_bias = getBNInfo(module)
                        ELAN_HNetInfo.append((name, (bn_weight, bn_mean, bn_var, bn_eps, bn_bias)))
                        continue

        if "RepConv" in i[0]:
           
            for name, module in i[1].named_children():
                if "rbr_dense" or "rbr_1x1" in name:
                    for name0, module0 in module.named_children():
                        module0Str = str(module0)
                      
                        if "Conv" in module0Str:
                            inChannel, outChannel, kernel_size, stride, padding, weight, bias = getConvInfo(module0)
                            ELAN_HNetInfo.append((name0, (inChannel, outChannel, kernel_size, stride, padding, weight, bias)))
                            continue
                        if "BatchNorm" in module0Str:
                            bn_weight, bn_mean, bn_var, bn_eps, bn_bias = getBNInfo(module0)
                            ELAN_HNetInfo.append((name0, (bn_weight, bn_mean, bn_var, bn_eps, bn_bias)))
                            continue
            
            
    return ELAN_HNetInfo

def getDectectInfo(Detectblock):
    DetectInfo = []

    for name, module in Detectblock.named_children():
        
        moduleStr = str(module)
        if "ModuleList" in moduleStr:
            for L0name, L0module in module.named_children():
                L0moduleStr = str(L0module)
               
                if "Conv2d" in L0moduleStr:
                    inChannel, outChannel, kernel_size, stride, padding, weight, bias = getConvInfo(L0module)
                    
                    DetectInfo.append((L0name, (inChannel, outChannel, kernel_size, stride, padding, weight, bias)))
                    
                    continue
            


    return DetectInfo                
