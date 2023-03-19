import os
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class UrbanSoundDataset(Dataset):
    def __init__(self, root_dir, file_list):
        self.root_dir = root_dir
        self.file_list = file_list
        self.num_classes = 10

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.root_dir, file_name)
        class_id = int(file_name.split("-")[1])
        waveform, sr = librosa.load(file_path, mono=True, sr=None)
        mel_spec = librosa.feature.melspectrogram(waveform, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec = np.expand_dims(mel_spec, axis=0)
        mel_spec = torch.from_numpy(mel_spec).float()

        label = np.zeros(self.num_classes)
        label[class_id] = 1.0
        label = torch.from_numpy(label).float()

        return mel_spec, label

# 训练集和测试集文件列表
train_file_list = []
test_file_list = []

for i in range(1, 11):
    fold_name = "fold" + str(i)
    fold_path = os.path.join("./UrbanSound8K", fold_name)
    file_list = os.listdir(fold_path)

    for file_name in file_list:
        if fold_name == "fold10":
            test_file_list.append(file_name)
        else:
            train_file_list.append(file_name)

# 数据集
train_dataset = UrbanSoundDataset("./UrbanSound8K", train_file_list)
test_dataset = UrbanSoundDataset("./UrbanSound8K", test_file_list)

# 数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("设备选择："+str(device))

# 模型
model = AudioClassifier(num_classes=10).to(device)

# 损失函数
criterion = nn.BCEWithLogitsLoss()

# 优化器
optimizer = torch.optim.Adam(model.parameters())

# 训练函数
def train(model, loader, criterion, optimizer):
    model.train()
    train_loss = 0.0
    train_acc = 0.0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        train_acc += (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().item()

    train_loss /= len(loader.dataset)
    train_acc /= len(loader.dataset)

    return train_loss, train_acc

# 测试函数
def test(model, loader, criterion):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            test_acc += (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().item()

    test_loss /= len(loader.dataset)
    test_acc /= len(loader.dataset)

    return test_loss, test_acc



best_acc = 0.0  # 用于记录最佳测试集准确率
num_epochs = 20
for epoch in range(num_epochs):
    # 训练并计算训练集损失和准确率
    train_loss, train_acc = train(model, train_loader, criterion, optimizer)

    # 在测试集上测试并计算测试集损失和准确率
    test_loss, test_acc = test(model, test_loader, criterion)

    # 打印统计信息
    print("Epoch: {:02d}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}".format(
        epoch + 1, train_loss, train_acc, test_loss, test_acc))

    # 如果当前测试集准确率超过之前的最佳测试集准确率，保存当前模型参数到文件中
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_model.pth')