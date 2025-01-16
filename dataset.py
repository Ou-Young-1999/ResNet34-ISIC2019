from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import os
import matplotlib.pyplot as plt

# 定义自己的数据集类
class CustomImageDataset(Dataset):
    def __init__(self, data_type, transform=None):
        self.img_dir = 'F:\\ISIC\\ISIC_2019_Training_Input\\ISIC_2019_Training_Input'
        self.data_type = data_type
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 将文件路径和标签存储在列表中
        txt_path = './preprocess/' + self.data_type + '.txt'
        with open(txt_path, 'r') as txt_file:
            for i, row in enumerate(txt_file):
                image_path = os.path.join(self.img_dir, row.split('\t')[0]+'.jpg')
                label = int(row.split('\t')[1])
                self.image_paths.append(image_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")  # 确保图像是RGB格式

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == '__main__':
    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])

    # 创建数据集实例
    train_dataset = CustomImageDataset(data_type='train', transform=transform)
    validation_dataset = CustomImageDataset(data_type='valid', transform=transform)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

    # 获取一个批次的数据
    data_iter = iter(validation_loader)
    images, labels = next(data_iter)

    gridImg = make_grid(images)
    print(labels)

    # 可视化网格图片
    plt.imshow(gridImg.permute(1, 2, 0))  # 调整通道顺序以适应 matplotlib 的要求
    plt.show()