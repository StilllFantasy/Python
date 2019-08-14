import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

BATCH_SIZE = 50
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([
    transforms.RandomResizedCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset_train = datasets.ImageFolder('D:\\DATA\\FISH_TRAIN - 副本 - 副本\\train_data\\', transform)
 
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=4, shuffle=True)
 
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
        # 展开
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
 

model = ConvNet().to(DEVICE)
# 选择简单暴力的Adam优化器，学习率调低
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 定义训练过程
def train(model, device, train_loader, optimizer, epoch):

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(output)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if(batch_idx+1)%10 == 0: 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                100. * (batch_idx+1) / len(train_loader), loss.item()))
    try:
        torch.save(model,'cnn-net11.pkl')
        torch.save(model,'cnn-net11.pth')
        print('保存神经网络成功')
    except:
        print('保存神经网络失败')

for epoch in range(0, 1025):
    train(model, DEVICE, train_loader, optimizer, epoch)