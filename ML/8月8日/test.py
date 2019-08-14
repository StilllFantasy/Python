import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
from PIL import ImageFile
import PIL
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True

transform = transforms.Compose([
    transforms.RandomResizedCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, 1, 2)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
        self.max_pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 5, 1, 2)
        self.max_pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(64, 128, 5, 1, 2)
        self.max_pool4 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(128, 256, 5, 1, 2)
        self.max_pool5 = nn.MaxPool2d(2)
        self.conv6 = nn.Conv2d(256, 256, 5, 1, 2)
        self.max_pool6 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 25)
    
    def forward(self, x):
        in_size = x.size(0)
        x = self.conv1(x)
        x = F.dropout(F.relu(x))
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.dropout(F.relu(x))
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = F.dropout(F.relu(x))
        x = self.max_pool3(x)
        x = self.conv4(x)
        x = F.dropout(F.relu(x))
        x = self.max_pool4(x)
        x = self.conv5(x)
        x = F.dropout(F.relu(x))
        x = self.max_pool4(x)
        x = self.conv6(x)
        x = F.dropout(F.relu(x))
        x = self.max_pool4(x)
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = F.dropout(F.relu(x))
        x = self.fc2(x)
        x = F.dropout(F.relu(x))
        x = self.fc3(x) 
        x = F.dropout(F.relu(x))
        x = self.fc4(x)
        x = F.dropout(F.relu(x))
        x = self.fc5(x)
        x = torch.log_softmax(x,dim=1)
        return x



def test(CLASS, inPATH, outPATH, NUM, net):
    val95 = []
    val90 = []
    val80 = []
    val70 = []
    namelist = os.listdir(inPATH)
    for idx, name in enumerate(namelist):
        print('[%d]' % CLASS,'计算', idx, len(namelist))
        filepath = inPATH + name
        try:
            image = PIL.Image.open(filepath)
        except:
            continue
        if image.getbands() != (('R', 'G', 'B')) :
            continue
        data = transform(image).unsqueeze(0).cuda()
        with torch.no_grad():
            out = net(data)
        value, index = out.topk(1, dim=1)
        value = 2**value
        if CLASS == index and value >= 0.95:
            val95.append(filepath)
        elif  CLASS == index and value >= 0.90:
            val90.append(filepath)
        elif  CLASS == index and value >= 0.80:
            val80.append(filepath)
        elif  CLASS == index and value >= 0.70:
            val70.append(filepath)
        else:
            continue
    vallist = val95 + val90 + val80 + val70
    for idx, imgfile in enumerate(vallist):
        if idx >= NUM:
            break
        print('[%d]' % CLASS,'处理', imgfile)
        image = PIL.Image.open(imgfile)
        savepath = outPATH + 'AFT' + imgfile[imgfile.rfind('\\')+1:] 
        try:
            image.save(savepath)
            print('[%d]' % CLASS,'保存', savepath)
        except:
            print('文件保存失败', savepath)
        try:
            os.remove(imgfile)
        except:
            print('文件删除失败')
        #a = input()

def run(netpath):
    net = torch.load(netpath)
    net.eval()
    for i in range(0, 25):
        CLASS = i
        inPATH = 'D:\\DATA\\FISH_TRAIN - 副本 - 副本\\unknown_data\\' + str(chr(ord('A')+i)) + '\\'
        outPATH = 'D:\\DATA\\FISH_TRAIN - 副本 - 副本\\train_data\\' + str(chr(ord('A')+i)) + '\\'
        print('正在筛选',CLASS)
        test(CLASS, inPATH, outPATH, 10, net)

def train(epoch):
    net = torch.load('D:\\Python\\cnn-net12(500_1024).pkl').cuda()
    dataset_train = datasets.ImageFolder('D:\\DATA\\FISH_TRAIN - 副本 - 副本\\train_data\\', transform)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=5, shuffle=True)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    net.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = net(data)
        loss = F.cross_entropy(output, target) 
        loss.backward()
        optimizer.step()
        if(batch_idx+1)%10 == 0: 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                100. * (batch_idx+1) / len(train_loader), loss.item()))
    try:
        torch.save(net,'D:\\Python\\cnn-net12(500_1024).pkl')
        print('保存神经网络成功')
    except:
        print('保存神经网络失败')

for t in range(40):
    train(t)
    train(t)
    run('D:\\Python\\cnn-net12(500_1024).pkl')
    