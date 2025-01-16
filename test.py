import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import CustomImageDataset
from model import resnet34
from tqdm import tqdm
import os
import random
import numpy as np
import csv
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed):
    random.seed(seed)  # 设置Python的随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    torch.manual_seed(seed)  # 设置PyTorch的CPU随机种子
    torch.cuda.manual_seed(seed)  # 设置当前GPU的随机种子（如果使用GPU）
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子（如果使用多个GPU）
    torch.backends.cudnn.deterministic = True  # 确保每次卷积操作结果一致
    torch.backends.cudnn.benchmark = False  # 禁用CUDNN的自动优化

# 训练函数
def test_model(model, test_loader, criterion, save_dir='./results'):
    # 创建模型保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()  # 设置模型为评估模式
    all_probabilities = []  # 存储所有样本的概率值保存到csv
    all_labels = []
    all_probs = []
    all_preds = []
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():  # 禁用梯度计算
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # 使用 softmax 函数获取类别概率
            probabilities = F.softmax(outputs, dim=1)

            # 将每个样本的概率值和其对应的标签保存
            for prob, label in zip(probabilities, labels):
                all_probabilities.append([label.item()] + prob.cpu().numpy().tolist())  # 转换为列表并保存

            # 计算损失
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probabilities.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())

            test_total += labels.size(0)

    # 保存为 CSV 文件
    with open(os.path.join(save_dir, 'prob.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Label'] + [f'Class_{i}' for i in range(probabilities.size(1))])  # 写入表头
        writer.writerows(all_probabilities)
    print(f"Probabilities saved to {os.path.join(save_dir, 'prob.csv')}")

    test_accuracy = 100 * test_correct / test_total
    test_loss = test_loss / len(test_loader)

    tqdm.write(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}%')

    # 拼接所有批次的标签和预测概率
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)

    return all_labels, all_probs, all_preds

def plot_roc_auc(all_labels, all_probs, n_classes, filename='roc_auc.png'):
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(f'Class {i} (AUC = {roc_auc[i]:.4f})')

    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.4f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(filename)
    plt.close()
    print(f"ROC saved to {filename}")

def plot_pr_auc(all_labels, all_probs, n_classes, filename='pr_auc.png'):
    precision, recall, pr_auc = {}, {}, {}
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(all_labels[:, i], all_probs[:, i])
        pr_auc[i] = auc(recall[i], precision[i])
        print(f'Class {i} (PR AUC = {pr_auc[i]:.4f})')

    plt.figure()
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], lw=2, label=f'Class {i} (PR AUC = {pr_auc[i]:.4f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig(filename)
    plt.close()
    print(f"PR saved to {filename}")


def plot_confusion_matrix(all_labels, all_preds, num_classes, filename='confusion_matrix.png'):
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 绘制热图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[f'Class {i}' for i in range(num_classes)],
                yticklabels=[f'Class {i}' for i in range(num_classes)])

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix saved to {filename}")

def compute_classification_report(all_labels, all_preds, num_classes, filename='report.txt'):
    report = classification_report(all_labels, all_preds, digits=4, target_names=[f'Class {i}' for i in range(num_classes)])
    with open(filename, 'w') as file:
        file.write(report)
    print(f"Report saved to {filename}")
    return report

if __name__ == '__main__':
    # 设置设备（GPU/CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} to test...')

    # 设置随机种子
    seed = 3407
    set_seed(seed)
    print(f'Random seed is {seed}')

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # 加载测试集
    test_dataset = CustomImageDataset(data_type='test', transform=transform)
    print(f'Testset size: {len(test_dataset)}')

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 导入模型
    num_classes = 8
    print(f'Loading model...')
    model = resnet34(num_classes)
    model = model.to(device)
    model.load_state_dict(torch.load('./models/best.pth'))

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    all_labels, all_probs, all_preds = test_model(model, test_loader, criterion)

    # 将标签二值化（适用于多类分类问题）
    all_labels_bin = label_binarize(all_labels, classes=[i for i in range(num_classes)])

    # 计算并保存 ROC AUC 曲线
    plot_roc_auc(all_labels_bin, all_probs, num_classes, filename='./results/roc_auc.png')

    # 计算并保存 PR AUC 曲线
    plot_pr_auc(all_labels_bin, all_probs, num_classes, filename='./results/pr_auc.png')

    # 输出混淆矩阵并保存为图片
    plot_confusion_matrix(all_labels, all_preds, num_classes, filename='./results/confusion_matrix.png')

    # 计算分类报告
    report = compute_classification_report(all_labels, all_preds, num_classes, filename='./results/report.txt')
    print(report)