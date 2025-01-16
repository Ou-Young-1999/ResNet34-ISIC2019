import math
import os
from PIL import Image

def merge_images(image_folder, output_folder, n, m, w, h):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有图像文件的列表
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]

    image_count = len(image_files)
    if image_count == 0:
        print('No image files found in the directory:', image_folder)
        return

    # 输出数据集中所有图片
    nums = math.ceil(image_count // (n*m))
    for num in range(nums):
        # 截取对应数量的图片
        image_files_split = image_files[n*m*num:n*m*(num+1)]

        # 设置小图像的大小
        img_size0 = w
        img_size1 = h
        new_img_size0 = img_size0 * n
        new_img_size1 = img_size1 * m

        # 创建一个新的大图像
        new_img = Image.new('RGB', (new_img_size0, new_img_size1), 'white')

        # 将所有小图像粘贴到新图像的正确位置
        for i, f in enumerate(image_files_split):
            row = int(i / n)
            col = i % n
            img = Image.open(f)
            img = img.resize((img_size0, img_size1))
            new_img.paste(img, (col * img_size0, row * img_size1))

        # 保存大图像
        num = str(num).zfill(3)  # 确保文件名长度相同
        new_img.save(os.path.join(output_folder,num+'.jpg'))

        print(f"Processed {image_folder}, saved to {output_folder}")

if __name__ == '__main__':
    # 用法示例
    image_folder = './results'
    output_file = './concat_result'
    n = 4  # 每行显示的图像数
    m = 5  # 每列显示的图像数
    w = 256*2  # 拼接小图宽度
    h = 256  # 拼接小图高度
    merge_images(image_folder, output_file, n, m, w, h)