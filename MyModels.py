from torch.utils.data import Dataset, DataLoader, random_split
import torch

#import scipy.io as sio
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset

from PIL import Image
import sys
import random


class C3D(nn.Module):
    def __init__(self):
        super(C3D, self).__init__()
        
        # pre-layer
        padding = (0, 3, 6, 0, 0, 0)
        self.padding = nn.ConstantPad3d(padding, 0.0)
        # 1st layer group
        self.conv1 = nn.Conv3d(12, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # 2nd layer group
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # 3rd layer group
        self.conv3a = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.relu3a = nn.ReLU()
        self.conv3b = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.relu3b = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # 4th layer group
        self.conv4a = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.relu4a = nn.ReLU()
        self.conv4b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.relu4b = nn.ReLU()
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # 5th layer group
        self.conv5a = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.relu5a = nn.ReLU()
        self.conv5b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.relu5b = nn.ReLU()
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # FC layers group
        self.fc6 = nn.Linear(512 * 4 * 4, 4096)
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU()
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(4096, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.padding(x)
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        
        x = self.relu3a(self.conv3a(x))
        x = self.relu3b(self.conv3b(x))
        x = self.pool3(x)
        
        x = self.relu4a(self.conv4a(x))
        x = self.relu4b(self.conv4b(x))
        x = self.pool4(x)
        
        x = self.relu5a(self.conv5a(x))
        x = self.relu5b(self.conv5b(x))
        x = self.pool5(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = self.dropout6(self.relu6(self.fc6(x)))
        x = self.dropout7(self.relu7(self.fc7(x)))
        x = self.softmax(self.fc8(x))
        
        return x

model = C3D()
print(model)
