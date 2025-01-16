import csv
import random

def read_csv(path):
    # 读取csv数据
    with open(path, 'r') as cvs_name:
        csv_reader = csv.reader(cvs_name)
        data = []
        for row in csv_reader:
            data.append(row)
    return data

def split_data(data):
    # 确保随机性
    random.seed(3407)
    random.shuffle(data)

    # 计算索引位置
    total_size = len(data)
    train_size = int(total_size * 0.6)
    val_size = int(total_size * 0.2)

    # 划分数据集
    train_set = data[:train_size]
    val_set = data[train_size:train_size + val_size]
    test_set= data[train_size + val_size:]

    return train_set, val_set, test_set

def write_txt(data, txt_path):
    image_and_label = [] # 保存图片名称和对应的标签
    category_statistics = [0,0,0,0,0,0,0,0] # 统计各类别数量
    for row in data:
        label = -1
        for i in range(1,len(row)):
            if row[i] == '1.0':
                label = i-1
                category_statistics[label] += 1
                break
        if label != -1:
            image_and_label.append(str(row[0])+'\t'+str(label))
        else:
            print(f'{row[0]} label is error!')
    print(f'{txt_path} Sategory Statistics: {category_statistics}')

    # 写入到txt
    with open(txt_path, 'w') as file:
        for case in image_and_label[:len(image_and_label)]:
            file.write(case + '\n')
        file.write(image_and_label[-1]) # 最后一行不需要换行

if __name__ == '__main__':
    csv_path = 'F:\\ISIC\\ISIC_2019_Training_GroundTruth.csv'
    data = read_csv(csv_path)[1:]
    print(f'total size: {len(data)}')
    train_set, val_set, test_set = split_data(data)
    print(f'train size: {len(train_set)}')
    print(f'valid size: {len(val_set)}')
    print(f'test size: {len(test_set)}')
    write_txt(train_set, 'train.txt')
    write_txt(val_set, 'valid.txt')
    write_txt(test_set, 'test.txt')