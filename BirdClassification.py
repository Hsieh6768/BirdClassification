import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import glob
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 数据集路径
image_dir = r"D:\Data\02_02_Office\25夏 计算机视觉导论\bird_photos"

# 图像参数
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_CLASSES = 20

# 数据预处理
base_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
])

# 训练数据增强
train_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 测试数据预处理
test_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class BirdDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # 获取所有类别文件夹
        self.class_folders = sorted([f for f in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, f))])
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.class_folders)}
        
        # 收集所有图像路径和标签
        self.image_paths = []
        self.labels = []
        
        for class_name in self.class_folders:
            class_image_dir = os.path.join(image_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # 获取图像文件
            image_files = [f for f in os.listdir(class_image_dir) if f.endswith('.jpg')]
            print(f"类别 {class_name} 有 {len(image_files)} 张图像")
            
            for image_file in image_files:
                image_path = os.path.join(class_image_dir, image_file)
                self.image_paths.append(image_path)
                self.labels.append(class_idx)
        
        print(f"总共加载了 {len(self.image_paths)} 张图像")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 定义ResNet的基本块
class IdentityBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(IdentityBlock, self).__init__()
        filters1, filters2, filters3 = filters
        
        self.conv1 = nn.Conv2d(in_channels, filters1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters1)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters2)
        
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filters3)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters, strides=2):
        super(ConvBlock, self).__init__()
        filters1, filters2, filters3 = filters
        
        self.conv1 = nn.Conv2d(in_channels, filters1, kernel_size=1, stride=strides, bias=False)
        self.bn1 = nn.BatchNorm2d(filters1)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters2)
        
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filters3)
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, filters3, kernel_size=1, stride=strides, bias=False),
            nn.BatchNorm2d(filters3)
        )
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += identity
        out = self.relu(out)
        
        return out

# 效仿ResNet50模型
class ResNetModel(nn.Module):
    def __init__(self, num_classes=200):
        super(ResNetModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 第一组块
        self.layer1 = nn.Sequential(
            ConvBlock(64, [64, 64, 256], strides=1),
            IdentityBlock(256, [64, 64, 256])
        )
        
        # 第二组块
        self.layer2 = nn.Sequential(
            ConvBlock(256, [128, 128, 512]),
            IdentityBlock(512, [128, 128, 512]),
            IdentityBlock(512, [128, 128, 512])
        )
        
        # 第三组块
        self.layer3 = nn.Sequential(
            ConvBlock(512, [256, 256, 1024]),
            IdentityBlock(1024, [256, 256, 1024]),
            IdentityBlock(1024, [256, 256, 1024]),
            IdentityBlock(1024, [256, 256, 1024])
        )
        
        # 第四组块
        self.layer4 = nn.Sequential(
            ConvBlock(1024, [512, 512, 2048]),
            IdentityBlock(2048, [512, 512, 2048]),
            IdentityBlock(2048, [512, 512, 2048])
        )

        # 第五组块
        self.layer5 = nn.Sequential(
            ConvBlock(2048, [1024, 1024, 4096]),
            IdentityBlock(4096, [1024, 1024, 4096])
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def main():
    # 创建数据集
    dataset = BirdDataset(image_dir, transform=base_transform)
    
    # 划分训练集和测试集
    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        random_state=42,
        stratify=dataset.labels
    )
    
     # 创建训练和测试数据集
    train_dataset = BirdDataset(image_dir, transform=train_transform)
    test_dataset = BirdDataset(image_dir, transform=test_transform)
    
    # 使用划分的索引创建子集
    train_dataset = Subset(train_dataset, train_indices)
    test_dataset = Subset(test_dataset, test_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetModel(num_classes=NUM_CLASSES)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # 打印模型概要
    print(model)
    
    # 训练模型
    num_epochs = 2
    
    # 训练历史记录容器
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': []
    }
    epoch_loss = 0.0
    train_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/(i+1):.4f}, Acc: {100.*correct/total:.2f}%')
        
        # 计算本 epoch 的训练损失与训练准确率
        epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        train_accuracy = 100. * correct / total if total > 0 else 0.0

        # 每个epoch结束后在测试集上评估
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_accuracy = 100. * test_correct / test_total
        scheduler.step(test_accuracy)  # 根据验证集性能调整学习率

        print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {test_accuracy:.2f}%')

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(train_accuracy)
        history['test_acc'].append(test_accuracy)
    

    # 训练损失
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.legend()

    # 训练/测试准确率
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['test_acc'], label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.close()
    
    
    # 计算并打印混淆矩阵
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 可视化混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=dataset.class_folders, 
                yticklabels=dataset.class_folders)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.close()

    
    # 保存模型
    torch.save(model.state_dict(), 'bird_classification_resnet44_20class.pth')
    print("模型已保存为 'bird_classification_resnet44_20class.pth'")


if __name__ == "__main__":
    main() 