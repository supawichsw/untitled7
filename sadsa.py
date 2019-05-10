
import time
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import glob
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib as plot
    plot.use("TkAgg")
from matplotlib import pyplot as plt
from PIL import Image
from  skimage import io,transform
import numpy as np
import math
from Network import CNN,train_load,train_dataset,test_dataset,test_load

features = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace=True),
    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2)
)

for m in features.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

