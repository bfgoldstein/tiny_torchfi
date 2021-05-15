import torch
import torch.nn as nn

from functools import reduce
from operator import getitem
import numpy as np

from .modules import *
from .bitflip import *
from util import *


class FI(object):

    def __init__(self, model, fiMode=False, fiBit=None, fiepoch=None, 
                 fiLayer=0, fiFeatures=True, fiWeights=True, log=False):
        self.model = model
        self.log = log
        
        self.injectionMode = fiMode
        self.injectionLayer = fiLayer
        self.injectionBit = fiBit
        self.injectionFeatures = fiFeatures
        self.injectionWeights = fiWeights
        self.numNewLayers = -1
        self.epoch = fiepoch

        self.numInjections = 0
        
        self.factory = {}
        self.fillFacotry()
        self.layersIds = []
        
    def traverseModel(self, model):
        for layerName, layerObj in model.named_children():
            newLayer = self.replaceLayer(layerObj, type(layerObj), layerName)
            setattr(model, layerName, newLayer)
            
            # For block layers we call recursively
            if self.has_children(layerObj):
                self.traverseModel(layerObj)
    
        return model
    
    def replaceLayer(self, layerObj, layerType, layerName):
        if self.injectionMode:
            if layerType in self.factory:
                return self.factory[layerType].from_pytorch_impl(self, layerName, layerObj)
        return layerObj

    def injectFeatures(self, tensorData, batchSize):
        faulty_res = []
        
        for batch_idx in range(0, batchSize):
            indices, faulty_val = self.inject(tensorData)
            faulty_res.append(([batch_idx] + indices, faulty_val))
        
        return faulty_res           

    def injectWeights(self, tensorData):
        return self.inject(tensorData)
    
    def inject(self, data: torch.Tensor):

        indices, data_val = getDataFromRandomIndex(data)
            
        if self.log:
            logInjectionNode("Node index:", indices)

        faulty_val, bit = bitFlip(data_val, bit=self.injectionBit, log=self.log) 
        
        self.injectionMode = False
        self.numInjections += 1
        
        return indices, faulty_val
    
    def setInjectionMode(self, mode):
        if self.log:
            logInjectionWarning("\tSetting injection mode to " + str(mode))
        self.injectionMode = mode

    def setInjectionBit(self, bit):
        if type(bit) == int and bit >= 0 and bit < 32:
            self.injectionBit = bit

    def setInjectionLayer(self, layer):
        self.injectionLayer = layer

    def has_children(self, module):
        try:
            next(module.children())
            return True
        except StopIteration:
            return False
    
    def addNewLayer(self, layerName, layerType):
        self.numNewLayers += 1
        self.layersIds.append((layerName, layerType))
        return self.numNewLayers

    def fillFacotry(self):
        self.factory[nn.Conv2d] = FIConv2d
        self.factory[nn.Linear] = FILinear
    
    def getInjectionLocation(self, size: int):
        return np.random.randint(size)

        
def getDataFromRandomIndex(data: torch.Tensor, random_state=None):
    # get item from multidimensional list at indices position 
    indices = [getRandomIndex(dim_size, random_state) for dim_size in data.shape]
    fiData = reduce(getitem, indices, data)
    return indices, fiData
    
    
def getRandomIndex(max: int, random_state=None):
    if random_state is not None:
        return random_state.randint(0, max)
    else:
        return np.random.randint(0, max)