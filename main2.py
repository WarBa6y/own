import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("CUDA存在"if torch.cuda.is_available() else '使用CPU')

# 定义模型
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(4*4*128, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


model = ConvNet().to(device)

# 定义超参数
batch_size = 100
num_epochs = 20
learning_rate = 0.001

# 加载数据集并移动到GPU上
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                 transform=transforms.ToTensor())
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
                                transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# 创建模型和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 可以使用StepLR调度器来每过一定的步数，将学习率乘以一个因子，来调整学习率
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 定义变量用于保存最佳正确率及其对应的模型参数
best_acc = 0.0
best_params = None

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):

    model.train()
    for i, (images, labels) in enumerate(train_loader):
        # 将数据移动到GPU上
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # 测试模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            # 将数据移动到GPU上
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print('Epoch [{}/{}], Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, acc))

    # 如果当前正确率大于最佳正确率，则保存当前模型参数
    if acc > best_acc:
        best_acc = acc
        best_params = model.state_dict()

    print('Updated Learning Rate: {:.6f}'.format(scheduler.get_last_lr()[0]))
# 保存最佳模型参数到文件中
torch.save(best_params, 'best_model.pth')