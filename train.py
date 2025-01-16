import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import CustomImageDataset
from model import resnet34
from tqdm import tqdm
import os
import random
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed):
    random.seed(seed)  # 设置Python的随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    torch.manual_seed(seed)  # 设置PyTorch的CPU随机种子
    torch.cuda.manual_seed(seed)  # 设置当前GPU的随机种子（如果使用GPU）
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子（如果使用多个GPU）
    torch.backends.cudnn.deterministic = True  # 确保每次卷积操作结果一致
    torch.backends.cudnn.benchmark = False  # 禁用CUDNN的自动优化

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, writer, num_epochs, save_dir='./models'):
    best_accuracy = 0.0

    # 创建模型保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        # 训练阶段
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as t:
            for inputs, labels in t:
                inputs, labels = inputs.to(device), labels.to(device)
                # 清零梯度
                optimizer.zero_grad()
                # 前向传播
                outputs = model(inputs)
                # 计算损失
                loss = criterion(outputs, labels)
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                # 更新进度条
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)
                accuracy = 100 * correct_preds / total_preds
                t.set_postfix(loss=running_loss / (t.n + 1), accuracy=accuracy)

        # 记录学习率
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch + 1)
        # 更新学习率
        scheduler.step()

        # 记录损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct_preds / total_preds

        # 验证阶段
        model.eval()  # 设置模型为评估模式
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():  # 禁用梯度计算
            for inputs, labels in tqdm(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)

        tqdm.write(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%')
        tqdm.write(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        writer.add_scalar('Loss/train', epoch_loss, epoch+1)
        writer.add_scalar('Accuracy/train', epoch_accuracy, epoch+1)

        writer.add_scalar('Loss/valid', val_loss, epoch+1)
        writer.add_scalar('Accuracy/valid', val_accuracy, epoch+1)

        # 保存最优模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
            tqdm.write(f"Best model saved with accuracy {best_accuracy:.2f}%")
        time.sleep(0.5)

if __name__ == '__main__':
    # 设置设备（GPU/CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} to train...')

    # 设置随机种子
    seed = 3407
    set_seed(seed)
    print(f'Random seed is {seed}')

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # 加载训练集和验证集
    train_dataset = CustomImageDataset(data_type='train', transform=transform)
    val_dataset = CustomImageDataset(data_type='valid', transform=transform)
    print(f'Trainset size: {len(train_dataset)}')
    print(f'Validationset size: {len(val_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 初始化SummaryWriter
    writer = SummaryWriter('runs/resnet34')
    writer.add_scalar('Loss/train', 10, 0)
    writer.add_scalar('Accuracy/train', 0, 0)
    writer.add_scalar('Loss/valid', 10, 0)
    writer.add_scalar('Accuracy/valid', 0, 0)

    # 创建模型
    model = resnet34(num_classes=8)
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # StepLR：每 10 个 epoch 衰减学习率
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 设置训练轮次
    num_epochs = 20

    # 训练模型
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, writer, num_epochs)

    writer.close()
