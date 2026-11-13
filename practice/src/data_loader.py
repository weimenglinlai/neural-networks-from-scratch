# data_loader.py
import os
import paddle
from PIL import Image
import paddle.vision.transforms as T
from paddle.io import Dataset, DataLoader

class FastTrashDataset(Dataset):
    """快速数据集类，优化了数据加载流程"""

    def __init__(self, data_dir, transform=None, mode='train', cache_size=1000):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.cache_size = cache_size

        # 自动检测类别
        self.classes = [d for d in os.listdir(data_dir)
                        if os.path.isdir(os.path.join(data_dir, d))]
        self.classes.sort()
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # 收集样本路径
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, self.class_to_idx[class_name]))

        print(f"加载 {mode} 数据集: {len(self.samples)} 张图片, {len(self.classes)} 个类别")

        # 预缓存部分数据
        self.image_cache = {}
        self._precache_images()

    def _precache_images(self):
        """预缓存部分图像到内存"""
        print(f"预缓存前 {min(self.cache_size, len(self.samples))} 张图像...")
        for i in range(min(self.cache_size, len(self.samples))):
            img_path, label = self.samples[i]
            try:
                image = Image.open(img_path).convert('RGB')
                self.image_cache[img_path] = image
            except Exception as e:
                print(f"预缓存失败: {img_path}, 错误: {e}")

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # 尝试从缓存获取图像
        if img_path in self.image_cache:
            image = self.image_cache[img_path]
        else:
            try:
                image = Image.open(img_path).convert('RGB')
                # 如果缓存未满，添加到缓存
                if len(self.image_cache) < self.cache_size:
                    self.image_cache[img_path] = image
            except Exception as e:
                print(f"无法加载图像: {img_path}, 错误: {str(e)}")
                # 返回黑色图像作为替代
                image = Image.new('RGB', (224, 224), color='black')
                label = 0

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.samples)

    def get_class_names(self):
        return self.classes

class PrefetchDataLoader:
    """预取数据加载器，在GPU计算时预加载下一批数据"""

    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        self.stream = paddle.device.cuda.Stream() if str(device).startswith('gpu') else None
        self.next_data = None
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.dataloader)
        except StopIteration:
            self.next_data = None
            return

        if self.stream is not None:
            with paddle.device.cuda.stream_guard(self.stream):
                self.next_data = [data.to(self.device, non_blocking=True)
                                  for data in self.next_data]

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_data is None:
            raise StopIteration

        current_data = self.next_data
        self.preload()
        return current_data

    def __len__(self):
        return len(self.dataloader)


def get_fast_transforms():
    """快速数据变换，减少预处理时间"""
    train_transform = T.Compose([
        T.Resize((256, 256)),  # 先调整到稍大尺寸
        T.RandomCrop((224, 224)),
        T.RandomHorizontalFlip(0.3),  # 减少数据增强强度
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = T.Compose([
        T.Resize((224, 224)),  # 验证集直接调整到目标尺寸
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def create_fast_data_loader(data_dir, batch_size=32, mode='train', use_prefetch=True):
    """创建快速数据加载器"""
    train_transform, val_transform = get_fast_transforms()
    transform = train_transform if mode == 'train' else val_transform

    dataset = FastTrashDataset(
        data_dir=data_dir,
        transform=transform,
        mode=mode,
        cache_size=500  # 缓存500张图像
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=0,  # Windows下设为0
        drop_last=(mode == 'train')
    )

    if use_prefetch and str(paddle.device.get_device()).startswith('gpu'):
        dataloader = PrefetchDataLoader(iter(dataloader), paddle.device.get_device())

    return dataloader, dataset