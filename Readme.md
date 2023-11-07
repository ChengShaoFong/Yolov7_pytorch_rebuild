# The Quantization model convert tool

## Environment Requirements
* numpy : 1.19
* torch : 1.7 or above
* torchvision : 1.7 or above
* matplotlib
* pillow

## Self-build model layers
* Definitions are under **./modelModules.py**.
* class convLayer(PruningModule).
* class fcLayer(PruningModule).
* class ResBlock(PruningModule).

## Get pre-trained model infos
* Definitions are under **./getInfo.py**.
* def getConvInfo(conv) : get convolution information.
* def getBNInfo(batchnorm) : get batch normalization information.
* def getFcInfo(module) : get fully-connected information.
* def getBiasBool(bias) : check bias or not.
* def getResBlockInfo(resblock) : get Resblock's information.

## Quantization Modules builder
* Definitions are under **./blockBuilder.py**.
* def getConvModule(convMod, bnMod) : build convolution layer module.
* def getFcModule(fcMod) : build fully-connected layer module.
* def getResBlock(resMod) : build ResBlock module.

## Main code
* Definitions are under **./modelConvert.py**.
* class modelBuilder(PruningModule) : the model builder
* def getModel(model) : this function does the model convertion
* def getData() : data loader is placed here
* def training(model, weightRoot, train_loader, test_loader, epochs=15) : for pretrained model training (only use for testing)
* def testing(model, test_loader) : test the model convert is success or not (accuracy before and after should be the same)