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

DEVICE = torch.device( 'cpu')

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
        x = torch.log_softmax(x, dim=1)
        return x


net = torch.load('D:\\Python\\cnn-net12(500_1024).pkl').cuda()


def REMOVE():
    removefilepath = os.listdir(str('C:\\Users\\user\\Desktop\\新建文件夹\\A\\'))
    for imagefile in removefilepath:
        removepath = str('C:\\Users\\user\\Desktop\\新建文件夹\\A\\') + str(imagefile)
        os.remove(removepath)


def screen(CLAS):
    # KEY = [0.60, 0.80, 0.90, 0.95, 1.00]
    # NUM = ['0.6', '0.8', '0.9', '0.95', '1.0']
    valuelist = [0.9, 1.0, 1.0, 1.0, 0.8, 0.8, 0.9, 0.6, 0.9, 0.8, 0.95, 0.8, 0.6, 1.0, 0.8, 0.8, 0.95, 0.95, 0.9, 1.0,
                 1.0, 0.8]
    # for idx, key in enumerate(KEY):
    readpath = 'D:\\DATA\\FISH_TRAIN - 副本 - 副本\\unknown_data\\' + str(CLAS) + str('\\')
    savepath = 'D:\\DATA\\FISH_TRAIN - 副本 - 副本\\intermediate_data\\' + str(CLAS) + str('\\')
    # keyvalue = str(NUM[idx])
    filepath = os.listdir(readpath)
    for idxx, filename in enumerate(filepath):
        # print('[%s][%s]' % (CLAS, keyvalue), idxx+1,'/',len(filepath))
        print(idxx, len(filepath))
        imagepath = readpath + filename
        try:
            image = PIL.Image.open(imagepath)
        except:
            continue
        if image.getbands() != (('R', 'G', 'B')):
            continue
        data = transform(image)
        data = data.unsqueeze(0).cuda()
        out = net(data)
        value, idx = out.topk(1, dim=1)
        value = 2 ** value
        if idx == int(ord(CLAS) - ord('A')) and value >= valuelist[int(ord(CLAS) - ord('A'))]:
            image.save(str(savepath + filename))


for t in range(0, 25):
    CLAS = chr(ord('A') + t)
    print('正在筛选', CLAS, '类鱼...')
    screen(CLAS)
    filepath = os.listdir('D:\\DATA\\FISH_TRAIN - 副本 - 副本\\intermediate_data\\' + str(CLAS) + str('\\'))
    ans = 0
    for idx, i in enumerate(filepath):
        ans += 1
    print(CLAS, '类鱼筛选完成！,共筛出：' , ans , '条')