# 导入库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import matplotlib.pyplot as plt 
import PIL
import numpy
from PIL import ImageFile
# 设置超参数
BATCH_SIZE = 50
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
# 数据预处理
transform = transforms.Compose([
    transforms.RandomResizedCrop(150),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
 
# 读取数据
# root = 'Cats_Dogs'
# dataset_train = datasets.ImageFolder('D:\\DATA\\Fish\\test', transform)
# dataset_test = datasets.ImageFolder('D:\\DATA\\Fish\\test', transform)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.max_pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.max_pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.max_pool4 = nn.MaxPool2d(2)
        self.conv5 = nn.Conv2d(128, 128, 3)
        self.max_pool5 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 25)
        
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
        # 展开
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = F.dropout(F.relu(x))
        x = self.fc2(x)
        x = F.dropout(F.relu(x))
        x = self.fc3(x) 
        x = torch.softmax(x,dim=1)
        return x
 

net = torch.load('D:\\Python\\cnn-net10.pkl')
net = net.cuda()


dataset_train = datasets.ImageFolder('D:\\比赛项目\\超算杯大数据比赛\\Fish_ImageData\\筛选数据\\', transform)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True)
ImageFile.LOAD_TRUNCATED_IMAGES = True
optimizer = optim.Adam(net.parameters(), lr=1e-4)

# def train():




def test1():
    
    true = 0
    for (image, label) in train_loader:
        image = image.cuda()
        with torch.no_grad():
            out = net(image)
        value, clas = out.topk(1, dim=1)
        label = label.cuda()
        label = label.unsqueeze(0)
        print('Act:', label, '   Pre:', clas, '   Probability:', float(value))
    
        if label == clas:
            true += 1

    print('Accuacy:', true,'/',len(train_loader))

def test2():
    net.eval()
    a = 0
    b = 0
    path = 'D:\\DATA\\Fish\\test\\D\\'
    for i in range(1,37):
        #print(i)
        filepath = path + str(i) + '.jpg'
        try:
            image = PIL.Image.open(filepath)
        except:
            print('ERROR')
            continue
        if image.getbands() == (('R', 'G', 'B')):
            # print('True')
            data = transform(image)
            data = data.unsqueeze(0)
            data = data.cuda()
            plt.imshow(image)
            with torch.no_grad():
                out = net(data)
            # print(image.size)
            value, clas = out.topk(1, dim=1)
            # value = 2**value
            name = round(float(value), 4)
            print(name)
            if clas == 3:
                a += 1
            else:
                b += 1 
    print(a, b)    

def test3(a, b, c, d, e, f, g, h, i, k):
    print('开始测试...')
    net.eval()
    imagedata = datasets.ImageFolder('D:\\DATA\\Fish\\噪声文件\\', transform)
    dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True)
    for idx, (image,  label) in enumerate(dataloader):
        print(idx, len(dataloader))
        image = image.cuda()
        with torch.no_grad():
            out = net(image)
        value, clas = out.topk(1, dim=1)
        name = round(float(value), 4)
        if name <= 0.1:
            a += 1 
        if name <= 0.2:
            b += 1
        if name <= 0.3:
            c += 1
        if name <= 0.4:
            d += 1
        if name <= 0.5:
            e += 1
        if name <= 0.6:
            f += 1
        if name <= 0.7:
            g += 1
        if name <= 0.8:
            h += 1
        if name <= 0.9:
            i += 1
    print('测试结束.')
    print(a, b, c, d, e, f, g, h, i)

test2()