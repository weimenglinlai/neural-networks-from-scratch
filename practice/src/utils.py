# utils.py
import os
import paddle
import paddle.vision.transforms as T
from paddle.io import Dataset
from PIL import Image

#数据集处理模块
class TrashDataset(Dataset):
    # 自动检测数据类别和构建数据集
    def __init__(self, data_dir, transform=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode

        # 自动检测类别
        self.classes = [d for d in os.listdir(data_dir)
                        if os.path.isdir(os.path.join(data_dir, d))]
        self.classes.sort()  # 确保顺序一致
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        img_path = os.path.join(class_dir, img_name)
                        # 检查文件是否可读
                        try:
                            with Image.open(img_path) as img:
                                img.verify()
                            self.samples.append((img_path, self.class_to_idx[class_name]))
                        except (IOError, SyntaxError) as e:
                            print(f"跳过损坏的图像: {img_path}")

        print(f"加载 {mode} 数据集: {len(self.samples)} 张图片, {len(self.classes)} 个类别")

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"无法加载图像: {img_path}, 错误: {str(e)}")
            # 返回一个黑色图像作为替代
            image = Image.new('RGB', (224, 224), color='black')
            label = 0

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.samples)

    def get_class_names(self):
        return self.classes


def get_transforms():
    """获取数据增强和预处理变换"""
    # 训练集使用数据增强
    train_transform = T.Compose([
        T.Resize((256, 256)),          # 先调整到稍大尺寸
        T.RandomCrop((224, 224)),      # 随机裁剪到目标尺寸
        T.RandomHorizontalFlip(0.5),   # 50%概率水平翻转
        T.RandomRotation(10),          # 随机旋转±10度
        T.ColorJitter(                 # 颜色抖动增强
            brightness=0.2,           # 亮度调整
            contrast=0.2,             # 对比度调整
            saturation=0.2,           # 饱和度调整
            hue=0.1                   # 色调调整
        ),
        T.ToTensor(),                 # 转换为张量
        T.Normalize(                  # 标准化
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 验证集只进行基本预处理
    val_transform = T.Compose([
        T.Resize((224, 224)),         # 直接调整到目标尺寸
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


def calculate_accuracy(outputs, labels):
    """计算准确率 - 修复数据类型问题"""
    predictions = paddle.argmax(outputs, axis=1)

    # 将 labels 转换为与 predictions 相同的数据类型
    if labels.dtype != predictions.dtype:
        labels = labels.astype(predictions.dtype)

    correct = (predictions == labels).astype('float32')
    accuracy = correct.sum() / len(correct)

    # 安全的获取准确率值
    try:
        accuracy_value = accuracy.numpy()
        if hasattr(accuracy_value, 'item'):
            return float(accuracy_value.item())
        else:
            return float(accuracy_value)
    except Exception as e:
        print(f"警告: 无法获取准确率: {e}")
        return 0.0


def load_model(model_class, model_path, num_classes=8):
    """加载训练好的模型"""
    model = model_class(num_classes=num_classes)
    if os.path.exists(model_path):
        state_dict = paddle.load(model_path)
        model.set_state_dict(state_dict)
        model.eval()
        return model
    else:
        raise FileNotFoundError(f"模型文件不存在: {model_path}")