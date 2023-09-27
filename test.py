
import torch
from tqdm.auto import tqdm
import itertools
import random
import logging
import pickle
from os.path import expanduser
import time
home = expanduser(~/Model_compression)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms, datasets
# Imports for plotting our result curves
import matplotlib
import matplotlib.pyplot as plt

def my_tester(model, valloader, length):
    val_acc = 0
    val_loss = 0
    model.eval()
    for j,input in enumerate(valloader,0):

        x = input[0].to(device)
        y = input[1].type(torch.LongTensor).to(device)

        
        out = model(x)

        loss = criterion(out,y)
        _, predicted = torch.max(out.data, 1)
        correct = (predicted == y).sum()

        val_acc += correct.item()
        val_loss += loss.item()

    val_acc /= length
    val_loss /= j
    return val_acc*100

class CIFAR3(Dataset):

    def __init__(self,transform=None):
      with open("cifar10_hst_test", 'rb') as fo:
          self.data = pickle.load(fo)
      
      self.transform = transform

    def __len__(self):
        return len(self.data['labels'])

    def __getitem__(self, idx):
        
        x = self.data['images'][idx,:]
        r = x[:1024].reshape(32,32)
        g = x[1024:2048].reshape(32,32)
        b = x[2048:].reshape(32,32)
        
        x = Tensor(np.stack([r,g,b]))

        if self.transform is not None:
          x = self.transform(x)
        
        y = self.data['labels'][idx,0]
        return x,y 

class SNet(nn.Module): #student
    def __init__(self):
        super(SNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.relu2 = nn.ReLU()
        # self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # self.relu3 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(1024, 512)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(512, 3)
        self.batchnorm1 = nn.BatchNorm1d(512)
       
    def forward(self, x):
        #TODO
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.pool1(self.conv2(x)))
        #print("#####################3", x.shape)
        #x = self.relu3(self.pool2(self.conv3(x)))
        nff = self.num_flat_features(x)
        x = x.view(-1 , nff)
        #print(x.shape, "###########")
        x = self.batchnorm1(self.fc1(x))
        x = self.relu5(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(4096, 512)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(512, 3)
        self.batchnorm1 = nn.BatchNorm1d(512)
       
    def forward(self, x):
        #TODO
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.pool1(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool2(x)
        #print(x.shape, "###########1")
        nff = self.num_flat_features(x)
        x = x.view(-1 , nff)
        #print(x.shape, "###########")
        x = self.batchnorm1(self.fc1(x))
        x = self.relu5(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

device = 'cpu'
model = SNet()
model_dict = torch.load(home+'/outputs/distil_prune.pth')
model.load_state_dict(model_dict['model_state_dict'])
criterion = torch.nn.CrossEntropyLoss()
test_transform = transforms.Compose([
        transforms.Normalize(mean=[127.5, 127.5, 127.5],
                             std=[127.5, 127.5, 127.5])
    ])

test_data = CIFAR3(transform=test_transform)
length = len(test_data)
batch_size = 256
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
tic = time.perf_counter()
valaccu = my_tester(model, testloader, length)
toc = time.perf_counter()
print(valaccu, " orig time ", toc - tic)
