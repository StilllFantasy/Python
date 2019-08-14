# 导入库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
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
dataset_train = datasets.ImageFolder('D:\\DATA\\人脸识别\\', transform)
dataset_test = datasets.ImageFolder('D:\\DATA\\Cat_Dog\\test_data', transform)
 
# 导入数据
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)
 
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
#model = ConvNet().to(DEVICE)
model = torch.load('D:\\Python\\ML\\Day11\\faceid.pkl').cuda()
#model = torch.load('D:\\Python\\cnn-net6.pkl')
# 选择简单暴力的Adam优化器，学习率调低
optimizer = optim.Adam(model.parameters(), lr=1e-4)
 
# 定义训练过程
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).float().reshape(1, 1)
        optimizer.zero_grad()
        output = model(data)
        # print(output)
        loss = F.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if(batch_idx+1)%10 == 0: 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                100. * (batch_idx+1) / len(train_loader), loss.item()))
    try:
        torch.save(model,'faceid.pkl')
        #torch.save(model,'cnn-net6.pth')
        print('保存神经网络成功')
    except:
        print('保存神经网络失败')


# 定义测试过程
def test(model, device, test_loader):
    print('正在测试...')
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).float().reshape(50, 1)
            output = model(data)
            # print(output)
            test_loss += F.binary_cross_entropy(output, target, reduction='sum').item() # 将一批的损失相加
            pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in output]).to(device)
            correct += pred.eq(target.long()).sum().item()
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
 
# 训练

for epoch in range(1, EPOCHS + 1): 
    train(model, DEVICE, train_loader, optimizer, epoch)