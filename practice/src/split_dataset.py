# split_dataset.py
import os
import shutil
import random


def split_dataset(data_root, output_dir, train_ratio=0.8, min_train_per_class=300, min_val_per_class=75):
    """
    划分数据集为训练集和验证集

    参数:
        data_root: 原始数据集根目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        min_train_per_class: 每个类别最少训练图片数
        min_val_per_class: 每个类别最少验证图片数
    """

    # 创建输出目录
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 获取所有类别
    classes = [d for d in os.listdir(data_root)
               if os.path.isdir(os.path.join(data_root, d))]

    print(f"找到 {len(classes)} 个类别: {classes}")

    for class_name in classes:
        class_dir = os.path.join(data_root, class_name)

        # 获取该类所有图片文件
        images = [f for f in os.listdir(class_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        print(f"类别 '{class_name}' 共有 {len(images)} 张图片")

        # 检查图片数量是否足够
        if len(images) < min_train_per_class + min_val_per_class:
            print(f"警告: 类别 '{class_name}' 图片数量不足!")
            continue

        # 随机打乱图片
        random.shuffle(images)

        # 确保每个类别至少有min_train_per_class张训练图片和min_val_per_class张验证图片
        train_images = images[:min_train_per_class]
        val_images = images[min_train_per_class:min_train_per_class + min_val_per_class]

        # 创建类别目录
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # 复制训练集图片
        for img in train_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(train_class_dir, img)
            shutil.copy2(src, dst)

        # 复制验证集图片
        for img in val_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(val_class_dir, img)
            shutil.copy2(src, dst)

        print(f"类别 '{class_name}': 训练集 {len(train_images)} 张, 验证集 {len(val_images)} 张")


def check_dataset(dataset_dir):
    """检查数据集划分结果"""
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')

    print("\n=== 数据集检查结果 ===")

    # 检查训练集
    train_classes = [d for d in os.listdir(train_dir)
                     if os.path.isdir(os.path.join(train_dir, d))]

    print(f"训练集包含 {len(train_classes)} 个类别:")
    for class_name in train_classes:
        class_dir = os.path.join(train_dir, class_name)
        images = [f for f in os.listdir(class_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        print(f"  {class_name}: {len(images)} 张图片")

    # 检查验证集
    val_classes = [d for d in os.listdir(val_dir)
                   if os.path.isdir(os.path.join(val_dir, d))]

    print(f"\n验证集包含 {len(val_classes)} 个类别:")
    for class_name in val_classes:
        class_dir = os.path.join(val_dir, class_name)
        images = [f for f in os.listdir(class_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        print(f"  {class_name}: {len(images)} 张图片")


if __name__ == "__main__":
    # 配置路径 - 请根据您的实际路径修改
    original_data_dir = "./data"  # 原始数据集路径
    output_data_dir = "./data"  # 划分后数据集输出路径

    print("开始划分数据集...")
    split_dataset(original_data_dir, output_data_dir)

    print("\n数据集划分完成!")
    check_dataset(output_data_dir)