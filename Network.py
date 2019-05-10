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

transform1=transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.Resize(64),transforms.ToTensor(),transforms.Normalize([0],[1])])


train_dataset = datasets.ImageFolder(root = 'Training',transform=transform1)

test_dataset = datasets.ImageFolder(root = 'Testing',transform = transform1)


#image_datasets = datasets.ImageFolder(root = 'Training',transform=transform1)
batch_size = 3
train_load = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = batch_size,
                                         shuffle = True,num_workers=0)

test_load=torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=0)
print(len(train_dataset))
print(train_dataset)
print(len(train_load))
print(train_load)

# get some random training images
images,labels = next(iter(train_load))
classname=train_dataset.classes
print(classname,"classname")
print(images)
print(labels,"labels")
sample_train_images = torchvision.utils.make_grid(images,nrow=4)
def imshow(inp):
    inp=inp
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0, 0, 0])
    std = np.array([1, 1, 1])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.pause(1)

imshow(sample_train_images)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(16)  # Batch normalization
        self.relu = nn.ReLU()  # RELU Activation
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # Maxpooling reduces the size by kernel size. 64/2 = 32

        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)  # Size now is 32/2 = 16

        # Flatten the feature maps. You have 32 feature mapsfrom cnn2. Each of the feature is of size 16x16 --> 32*16*16 = 8192
        self.fc1 = nn.Linear(in_features=8192,
                             out_features=4000)  # Flattened image is fed into linear NN and reduced to half size
        self.droput = nn.Dropout(p=0.5)  # Dropout used to reduce overfitting
        self.fc2 = nn.Linear(in_features=4000, out_features=2000)
        self.droput = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=2000, out_features=500)
        self.droput = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(in_features=500, out_features=50)
        self.droput = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(in_features=50,
                             out_features=2)  # Since there were so many features, I decided to use 45 layers to get output layers. You can increase the kernels in Maxpooling to reduce image further and reduce number of hidden linear layers.

    def forward(self, x):
        out = self.cnn1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        # Flattening is done here with .view() -> (batch_size, 32*16*16) = (100, 8192)
        out = out.view(-1, 8192)  # -1 will automatically update the batchsize as 100; 8192 flattens 32,16,16
        # Then we forward through our fully connected layer
        out = self.fc1(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc5(out)
        return out
