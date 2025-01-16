import sys
import os
import cv2
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import resnet34
import torch
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
def batch_process_gradcam(model, image_paths, target_layer, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    for image_path in image_paths:
        # 加载并预处理图像
        image = Image.open(image_path).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])

        input_tensor = transform(image).unsqueeze(0)  # 增加批次维度

        # 获取类别的预测
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
        predicted_class = predicted_class.cpu().numpy()[0]

        # Grad-Cam算法
        cam = GradCAM(model=model, target_layers=target_layer)
        grayscale_cam = cam(input_tensor=input_tensor)

        # 取第1张图的cam
        grayscale_cam = grayscale_cam[0, :]

        # 将CAM作为掩码(mask)叠加到原图上
        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (256,256))
        rgb_img = np.float32(rgb_img) / 255
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        rgb_img = (rgb_img * 255).astype(np.uint8)
        img_hstack = np.hstack([rgb_img, cam_image])

        # 可视化并保存热图
        image_file = os.path.basename(image_path)
        output_image_path = os.path.join(output_folder, f"gradcam_{predicted_class}_{image_file}")
        Image.fromarray((img_hstack).astype(np.uint8)).save(output_image_path)
        print(f"Processed {image_file}, saved to {output_image_path}")

if __name__ == '__main__':
    # 使用模型
    print(f'Loading model...')
    model = resnet34(num_classes=8)
    model = model.to(device='cpu')
    model.load_state_dict(torch.load('../models/best.pth'))
    model.eval()

    # 选择 ResNet50 的最后一个卷积层（layer4 的最后一个卷积块）
    target_layer = [model.layer4[-1].conv2]

    # 获取图像文件路径
    img_dir = 'F:\\ISIC\\ISIC_2019_Training_Input\\ISIC_2019_Training_Input'
    image_paths = []
    labels = []
    with open('../preprocess/test.txt', 'r') as txt_file:
        for i, row in enumerate(txt_file):
            image_path = os.path.join(img_dir, row.split('\t')[0] + '.jpg')
            label = int(row.split('\t')[1])
            image_paths.append(image_path)
            labels.append(label)
    output_folder = 'results'

    # 批量处理图像并保存 Grad-CAM 热图
    batch_process_gradcam(model, image_paths[0:20], target_layer, output_folder)
