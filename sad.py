# Training the CNN

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
from Network import CNN,train_load,train_dataset,test_dataset,test_load

model = CNN()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device,"device")
CUDA = torch.cuda.is_available()
if CUDA:
    model = model.cuda()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

num_epochs = 50

# Define the lists to store the results of loss and accuracy
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

# Training
for epoch in range(num_epochs):
    # Reset these below variables to 0 at the begining of every epoch
    start = time.time()
    correct = 0
    iterations = 0
    iter_loss = 0.0

    model.train()  # Put the network into training mode

    for i, (inputs, labels) in enumerate(train_load):

        # Convert torch tensor to Variable
        inputs = Variable(inputs)
        labels = Variable(labels)

        CUDA = torch.cuda.is_available()
        if CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()  # Clear off the gradient in (w = w - gradient)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        iter_loss += loss.item() * inputs.size(0)  # Accumulate the loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update the weights

        # Record the correct predictions for training data
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum()
        iterations += 1

    # Record the training loss
    train_loss.append(iter_loss / iterations)
    # Record the training accuracy
    train_accuracy.append((100 * correct / len(train_dataset)))

    # Testing
    loss = 0.0
    correct = 0
    iterations = 0

    model.eval()  # Put the network into evaluation mode

    for i, (inputs, labels) in enumerate(test_load):

        # Convert torch tensor to Variable
        inputs = Variable(inputs)
        labels = Variable(labels)

        CUDA = torch.cuda.is_available()
        if CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()



        outputs = model(inputs)
        loss = loss_fn(outputs, labels)  # Calculate the loss
        loss += loss.item() * inputs.size(0)
        # Record the correct predictions for training data
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum()

        iterations += 1

        # Record the Testing loss
    test_loss.append(loss / iterations)
    # Record the Testing accuracy
    test_accuracy.append((100 * correct / len(test_dataset)))
    stop = time.time()

    print(
        'Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Testing Loss: {:.3f}, Testing Acc: {:.3f}, Time: {}s')


