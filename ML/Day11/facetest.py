# 导入库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import matplotlib.pyplot as plt 
import os
from PIL import Image
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
root = 'Cats_Dogs'
dataset_train = datasets.ImageFolder('D:\\DATA\\Cat_Dog\\train_data', transform)
dataset_test = datasets.ImageFolder('D:\\DATA\\Cat_Dog\\train_data', transform)

# 导入数据
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=True)
 
# 定义网络
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
        self.fc1 = nn.Linear(6272, 512)
        self.fc2 = nn.Linear(512, 1)
        
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
        # 展开
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = F.dropout(F.relu(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        
        return x
 
# 实例化模型并且移动到GPU
# model = ConvNet().to(DEVICE)
# 选择简单暴力的Adam优化器，学习率调低
# optimizer = optim.Adam(model.parameters(), lr=1e-4)

num = 0

model = torch.load('D:\\Python\\ML\\Day11\\faceid.pkl').cuda()
#model = torch.load('D:\\Python\\cnn-net6.pkl')
#for idx, (image, label) in enumerate(dataset_test):
#    image = image.unsqueeze(0).cuda()
#    with torch.no_grad():
#        out = model(image)
#    pre = 1 if out >=0.5 else 0
#    if pre == label:
#        num += 1
#    print(idx)
#print(num, len(dataset_test))
mnhvmh gj,gkuiyt7ui,ytfgj
path = 'D:\\DATA\\人脸识别测试\\谁更像\\'
namelist = os.listdir(path)
for i in namelist:
    filepath = path + i
    image = Image.open(filepath)
    data = transform(image)
    data = data.unsqueeze(0).cuda()
    with torch.no_grad():
        out = model(data)
    pre = 1 if out >= 0.5 else 0
    prename = ['比尔·盖茨', '史蒂夫·乔布斯']
    print('[%s]长得更像[%s] value = '%(i[:i.find('.')], prename[pre]), float(out))

'''
for i in range(1002,1020):
    filepath1 = 'D:\\DATA\\Cat_Dog\\test_data\dog\\' + str(i) + '.jpg' # 1002~1501
    filepath2 = 'D:\\DATA\\Cat_Dog\\test_data\cat\\' + str(i) + '.jpg' # 1002~1501
    filepath3 = 'D:\\Python\\DownloadCat\\Pictures\\' + str(i) + '.jpg'  # 1~19
    filepath4 = 'D:\\DATA\\Fish\\A\\' + str(i) + '.jpg'                  # 1~39
    image = Image.open(filepath1)
    plt.imshow(image)
    data = transform(image)
    data = data.unsqueeze(0)
    data = data.cuda()
    out = model(data)
    prediction = 'dog' if out >= 0.5 else 'cat'
    print(i,prediction)
    plt.show()
'''