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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def REMOVE():
    removefilepath = os.listdir(str('C:\\Users\\user\\Desktop\\新建文件夹\\A\\'))
    for imagefile in removefilepath:
        removepath = str('C:\\Users\\user\\Desktop\\新建文件夹\\A\\') + str(imagefile)
        os.remove(removepath)

def test(path):
    net = torch.load(path).cuda()
    num = 0 
    total = 0
    path = 'D:\\DATA\\新建文件夹 (2)\\'
    namelist = os.listdir(path)
    for i in namelist:
        filename = path + i
        image = PIL.Image.open(filename)
        data = transform(image)
        data = data.unsqueeze(0).cuda()
        with torch.no_grad():
            out = net(data)
        value, idex = out.topk(1,dim=1)
        value = round(float(value), 5)
        # print('[%s]:[%s]' % (i[0], chr(int(idex)+ord('A'))))
        total += 1
        if i[0] == chr(int(idex)+ord('A')):
            num += 1
    print('正确张数：', num, '/', total)
    print('正确率：%.4f%%' % (num/total))    
    print()

def screen(CLAS, path):
    net = torch.load(path).cuda()
    
    KEY = [0.60, 0.80, 0.90, 0.95, 1.00]
    NUM = ['0.60', '0.80', '0.90', '0.95', '1.00']
    for idx, key in enumerate(KEY):
        key = round(key,2)
        readpath = 'D:\\DATA\\FISH_TRAIN - 副本 - 副本\\unknown_data\\' + str(CLAS) + str('\\')
        savepath = 'C:\\Users\\user\\Desktop\\新建文件夹\\' + str(CLAS) + str('\\') + str(NUM[idx]) + str('\\')
        keyvalue = str(NUM[idx])
        filepath = os.listdir(readpath)
        for idxx, filename in enumerate(filepath):
            print('[%s][%s]' % (CLAS, keyvalue), idxx+1,'/',len(filepath))
            imagepath = readpath + filename
            try:
                image = PIL.Image.open(imagepath)
            except:
                continue
            if image.getbands() != (('R', 'G', 'B')) :
                continue
            data = transform(image)
            data = data.unsqueeze(0).cuda()
            out = net(data)
            value, idx = out.topk(1, dim=1)
            value = 2**value
            if idx == int(ord(CLAS) - ord('A')) and value >= key:
                image.save(str(savepath+filename))

def test2(path):
    num = 0
    net = torch.load(path).cuda()
    data_set = datasets.ImageFolder('D:\\DATA\\FISH_TRAIN - 副本 - 副本\\unknown_data\\', transform)
    data_load = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=True)
    lenn = len(data_load)
    for number, (image, label) in enumerate(data_load):
        image = image.cuda()
        label = label.cuda()
        with torch.no_grad(): 
            out = net(image)
        value, indx = out.topk(1, dim=1)
        if indx == label :
            num += 1
        print(number,'/',lenn)
        if number % 100 == 0 and number != 0:
            print(num, num/number)
    print('测试完成')

def test3(path):
    net = torch.load(path).cuda()
    net.eval()
    path = 'C:\\Users\\user\\Desktop\\筛出待统计\\test\\'
    filelist = os.listdir(path)
    yes = 0
    no = 0
    lenn = len(filelist)
    for num, name in enumerate(filelist):
        filepath = path + name
        try:
            image = PIL.Image.open(filepath)
        except:
            continue
        if image.getbands() != (('R', 'G', 'B')) :
                continue
        data = transform(image)
        data = data.unsqueeze(0).cuda()
        with torch.no_grad():
            out = net(data)
        value, idx = out.topk(1,dim=1)
        value = 2**float(value)
        print('[%d:%d][%s]' % (num, lenn, name), end=' ')
        print('[%.4f] [%d]' % (value, idx), end=' ')
        if value < 0.5 :
            print('[N]')
            no += 1
        else :
            print('[Y]')
            yes += 1
    
    print('比例为：', yes, no)

path1 = 'D:\\Python\\Net_Model\\cnn-net12(500_1024).pkl'
path2 = 'D:\\Python\\Net_Model\\cnn-net12(500_1024) copy.pkl'
path3 = 'D:\\Python\\Net_Model\\net1-400.pkl'
test3(path3)