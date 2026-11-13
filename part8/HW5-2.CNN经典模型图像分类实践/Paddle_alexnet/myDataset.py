# -*- coding:utf-8 -*-
"""
作者：自福
日期：2024年04月17日
"""

import os
from paddle.vision import transforms
from paddle.io import Dataset
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class MyDataset(Dataset):   # 该类中的__getitem__方法采用 PIL 读取图像，并返回图像和对应标签
    def __init__(self, data_dir, transform=None):
        """
        1.继承 paddle.io.Dataset 类
        2.实现 __init__ 函数，初始化数据集，将样本和标签映射到列表中
        """
        super(MyDataset, self).__init__()
        self.classes = []
        self.class_to_idx = {}
        self.data_list = []
        # 获取data_dir目录下的所有文件夹（即所有类别），将文件夹名称存入名为class_list的列表中，即['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
        self.classes = [f.name for f in os.scandir(data_dir) if f.is_dir()]
        # 以class_list中的数据项的索引为key，数据项为value构建一个class_dict的字典，即{'daisy':0, 'dandelion':1, 'rose':2, 'sunflower':3, 'tulip':4}
        self.class_to_idx = {folder: index for index, folder in enumerate(self.classes)}

        # 使用os.walk()遍历目录及其子目录，获取所有文件的绝对路径
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                # 检查文件后缀名是否为.jpg或.png
                if file.lower().endswith(('.jpg','.jpeg','.png','.bmp','.gif','.tif','.tiff','.ppm','.pgm')):
                    # 构建完整的文件路径并添加到file_list中
                    image_path = os.path.join(root, file)
                    class_name = image_path.split(os.path.sep)[-2]  # 从图像文件绝对路径中获取类别名称
                    label = self.class_to_idx[class_name]  # 从类别名称转换为类别标签
                    self.data_list.append([image_path, label])  # 将图像路径和对应的类别标签添加到data_list中

        # 传入定义好的数据处理方法，作为自定义数据集类的一个属性
        self.transform = transform

    def __getitem__(self, index):
        """
        3.实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
        """
        # 确保 index 在有效范围内
        if index >= len(self.data_list):
            raise IndexError("Index out of dataset range")

        # 根据索引，从列表中取出一个图像
        image_path, label = self.data_list[index]
        # 初始化变量用于存储图像数据
        image = None

        # 尝试读取图像，如果失败则重新尝试直到成功
        while image is None:
            # 使用 PIL 读取图像
            try:
                with Image.open(image_path) as img:
                    image = img.convert('RGB')  # 确保图像是 RGB 格式
                    image = np.array(image)
            except IOError as e:
                # 如果图像读取失败，打印错误并返回 None
                print(f"无法读取图像文件 {image_path}: {e}")
                # 这里可以添加逻辑来选择跳过当前图像或者从头开始重新尝试
                # 例如，可以选择跳过当前图像并使用下一个图像：
                # index += 1
                # 如果 index 超出数据集范围，则从头开始：
                # if index >= len(self.data_list):
                #     index = 0
                # image_path, label = self.data_list[index]


        # 飞桨训练时内部数据格式默认为float32，将图像数据格式转换为 float32
        image = image.astype('float32')
        # print(image)
        # 应用数据处理方法到图像上
        if self.transform is not None:
            image = self.transform(image)
            # print('transformed image:', image)
        # CrossEntropyLoss要求label格式为int，将Label格式转换为 int
        label = int(label)
        # 返回图像和对应标签
        return image, label

    def __len__(self):
        """
        4.实现 __len__ 函数，返回数据集的样本总数
        """
        return len(self.data_list)

# 该类中的__getitem__方法采用cv2.imread()读取图像，并返回图像和对应标签
# 注意：cv2.imread()读取图像时不支持中文路径
class MyDataset1(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        1.继承 paddle.io.Dataset 类
        2.实现 __init__ 函数，初始化数据集，将样本和标签映射到列表中
        """
        super(MyDataset1, self).__init__()
        self.classes = []
        self.class_to_idx = {}
        self.data_list = []
        # 获取data_dir目录下的所有文件夹（即所有类别），将文件夹名称存入名为class_list的列表中，即['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
        self.classes = [f.name for f in os.scandir(data_dir) if f.is_dir()]
        # 以class_list中的数据项的索引为key，数据项为value构建一个class_dict的字典，即{'daisy':0, 'dandelion':1, 'rose':2, 'sunflower':3, 'tulip':4}
        self.class_to_idx = {folder: index for index, folder in enumerate(self.classes)}

        # 使用os.walk()遍历目录及其子目录，获取所有文件的绝对路径
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                # 检查文件后缀名是否为.jpg或.png
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.ppm', '.pgm')):
                    # 构建完整的文件路径并添加到file_list中
                    image_path = os.path.join(root, file)
                    class_name = image_path.split(os.path.sep)[-2]  # 从图像文件绝对路径中获取类别名称
                    label = self.class_to_idx[class_name]  # 从类别名称转换为类别标签
                    self.data_list.append([image_path, label])  # 将图像路径和对应的类别标签添加到data_list中

        # 传入定义好的数据处理方法，作为自定义数据集类的一个属性
        self.transform = transform

    def __getitem__(self, index):
        """
        3.实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
        """
        # 根据索引，从列表中取出一个图像
        image_path, label = self.data_list[index]
        # 读取图像（不支持中文路径）
        # image = cv2.imread(image_path)

        #  读取图像（支持中文路径）
        # 使用 np.fromfile() 读取图像文件到内存缓冲区
        raw_data = np.fromfile(image_path, dtype=np.uint8)
        # 使用 cv2.imdecode() 将缓冲区解码成图像
        image = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)

        # 将BGR格式的图像转换为RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 飞桨训练时内部数据格式默认为float32，将图像数据格式转换为 float32
        image = image.astype('float32')
        # 应用数据处理方法到图像上
        if self.transform is not None:
            image = self.transform(image)
        # CrossEntropyLoss要求label格式为int，将Label格式转换为 int
        label = int(label)
        # 返回图像和对应标签
        return image, label

    def __len__(self):
        """
        4.实现 __len__ 函数，返回数据集的样本总数
        """
        return len(self.data_list)


if  __name__ == '__main__':
    # 定义数据预处理转换操作
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 随机大小裁剪，输出大小为224x224
                                     transforms.RandomHorizontalFlip(),  # 随机水平翻转
                                     transforms.ToTensor(),  # 将PIL图像或NumPy数组转换为FloatTensor
                                     transforms.Normalize(mean = [127.5, 127.5, 127.5], std = [127.5, 127.5, 127.5])]),  # 归一化
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)  调整图像大小为224x224
                                   transforms.ToTensor(),  # 转换为Tensor
                                   transforms.Normalize(mean = [127.5, 127.5, 127.5], std = [127.5, 127.5, 127.5])])}  # 归一化

    # 获取上两级目录的绝对路径作为数据集的根目录
    data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))  # get data root path，“../..”是上2级目录，".."是上1级目录
    image_path = os.path.join(data_root, "data_set", "flowers_data")  # 定义图像数据集路径
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)  # 确保图像数据集路径存在

    # 定义数据集并打印数据集样本数
    train_custom_dataset = MyDataset(data_dir=os.path.join(image_path, "train"), transform=data_transform["train"])
    val_custom_dataset = MyDataset(data_dir=os.path.join(image_path, "val"), transform=data_transform["val"])
    print('train_custom_dataset images: ', len(train_custom_dataset), '\n','test_custom_dataset images: ',
          len(val_custom_dataset))

    # 迭代读取数据集
    for data in val_custom_dataset:
        image, label = data
        # 首先将其转换为 NumPy 数组，并确保通道顺序为 (height, width, channels)
        image = np.array(image).transpose((1, 2, 0))
        # 确保图像数据的数据类型为 uint8
        # image = np.clip(image, 0, 255).astype(np.uint8)
        # 对图像数据进行归一化，使其范围在 [0, 1]
        # image = image.astype(np.float32) / 255.0
        print('shape of image: ', image.shape)
        plt.title(str(label))
        plt.imshow(image)
        plt.show()
        # 暂停并等待用户输入，以便逐张查看图片
        input("Press Enter to continue...")
        # break