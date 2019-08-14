import PIL
import matplotlib.pyplot as plt 
import os
import random
import torch
from torchvision import datasets, transforms
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
transform = transforms.Compose([
    transforms.RandomResizedCrop(150),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
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
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = F.dropout(F.relu(x))
        x = self.fc2(x)
        x = F.dropout(F.relu(x))
        x = self.fc3(x) 
        x = torch.log_softmax(x,dim=1)
        return x

Net = torch.load('D:\\Net_Model\\25_fishes_classfier.pkl')
optimizer = optim.Adam(Net.parameters(), lr=1e-4)

def train(T=0):
    global Net
    Net = torch.load('cnn-net10.pkl')
    print('开始第%d次训练...' % T)
    Net.train()
    Net = Net.cuda()
    dataset_train = datasets.ImageFolder('D:\\DATA\\FISH_TRAIN\\train_data\\', transform)
    data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True)
    optimizer = optim.Adam(Net.parameters(), lr=1e-4)
    # model.train() 
    for epoch in range(10):
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = Net(data)
            # print(output)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if(batch_idx+1)%10 == 0: 
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx+1) * len(data), len(data_loader.dataset),
                    100. * (batch_idx+1) / len(data_loader), loss.item()))
        try:
            torch.save(Net,'cnn-net10.pkl')
            torch.save(Net,'cnn-net10.pth')
            print('保存神经网络成功')
        except:
            print('保存神经网络失败')


def screen(T=0):
    global Net
    Net = torch.load('cnn-net10.pkl')
    print('正在进行第%d次筛选图片...' % T)
    Net.eval()
    # Net = torch.load('D:\\Net_Model\\25_fishes_classfier.pkl')
    keylist = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    for key in range(0,25):
        path = 'D:\\DATA\\FISH_TRAIN\\unknown_data\\' + str(keylist[key]) + '\\'
        print('正在筛选',str(keylist[key]),'类图片:')
        namelist = os.listdir(path)
        for numbers, name in enumerate(namelist):
            filepath = path + str(name)
            try:
                image = PIL.Image.open(filepath)
            except:
                continue
            if image.getbands() != (('R', 'G', 'B')):
                continue
            data = transform(image)
            data = data.unsqueeze(0).cuda()
            with torch.no_grad():
                out = Net(data)
            value, index = out.topk(1, dim=1)
            value = round(float(value), 4)
            value = 2**value
            if value <= 0.25:
                oslen = len(os.listdir('D:\\DATA\\FISH_TRAIN\\noise_data\\'))
                remove_path = 'D:\\DATA\\FISH_TRAIN\\noise_data\\' + str(oslen) + '.jpg'
                try:
                    image.save(remove_path)
                    os.remove(filepath)
                    print(numbers,'/',len(namelist),'已成功剔除', str(keylist[key]), '噪声图片：',filepath)
                except:
                    print(numbers,'/',len(namelist),'剔除失败！')
            elif value >= 0.8 and index == key:
                savepath = 'D:\\DATA\\FISH_TRAIN\\intermediate_data\\' + str(keylist[key]) + '\\' + str(name)
                try:
                    image.save(savepath)
                    print(numbers,'/',len(namelist),'已保存', str(keylist[key]), '中间图片：', savepath)
                except:
                    print(numbers,'/',len(namelist),'保存失败！')
        path = 'D:\\DATA\\FISH_TRAIN\\intermediate_data\\' + str(keylist[key]) + '\\'
        namelist = os.listdir(path)
        save_numbers = 0
        oslen = int(len(namelist)/3)
        
        while True:
            if save_numbers >= oslen:
                break
            oslen_1 = len(namelist)
            index = random.randint(0, oslen_1)
            
            try: 
                image = PIL.Image.open(str(path) + str(namelist[index]))
                image.save('D:\\DATA\\FISH_TRAIN\\train_data\\' + str(keylist[key]) + '\\' + str(namelist[index]))
                print(save_numbers,'/',oslen, '成功随机保存一张',str(keylist[key]), str(namelist[index]))
                save_numbers += 1
            except:
                print(save_numbers,'/',oslen, '随机保存失败！')
        
        print('本次筛出', str(keylist[key]), '图片', save_numbers, '张.')

        for image in namelist:
            imagepath = path + image
            try:
                os.remove(imagepath)
            except:
                continue
        print('已成功清除', str(keylist[key]), '类图片的中间数据.')
        
    print('已成功筛选所有类别，下面进行训练.')


def run():
    for t in range(10):
        train(t)
        screen(t)  

run()
