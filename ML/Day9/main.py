import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt 
import numpy as np 
from PIL import Image


# Load the image data and transform to Tensor
transform = transforms.Compose([transforms.RandomResizedCrop(150),transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
data = datasets.ImageFolder('D:\\DATA\\Cat_Dog\\test_data', transform)
testdata = datasets.ImageFolder('D:\\DATA\\Cat_Dog\\test_data', transform)


# data = data[0:1000]
dataloader = DataLoader(data, batch_size=1, shuffle=True)


# print(data.class_to_idx)
# print(data[1][0].size())

# define the cnn neural network


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.max_pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.max_pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.max_pool4 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(6272, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        in_size = x.size(0)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.max_pool3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max_pool4(x)
        # 展开
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x


net = Net().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_fac = torch.nn.CrossEntropyLoss()


def Train():
    print('Start training...')
    for step in range(10):
        for t, (data, target) in enumerate(dataloader): 
            print(step, ':', t, '/', len(dataloader))
            data = data.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            output = net(data)
            loss = loss_fac(output, target)
            loss.backward()
            optimizer.step()
    
    print('training ended.')
    torch.save(net, 'cnn-net4')

def Test():
    testloader = DataLoader(testdata,batch_size=1,shuffle=False)
    
    for t, (data, target) in enumerate(testloader):
        data = data.cuda()
        with torch.no_grad():
            out = net(data)
        print(out,' ', target)


Train()
Test()