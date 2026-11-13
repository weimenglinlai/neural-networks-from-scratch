import os
import sys
import json
import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import Dataset, DataLoader
from paddle.vision import transforms, datasets
import matplotlib.pyplot as plt
from tqdm import tqdm
import paddle.optimizer as optim

from myDataset import MyDataset

from model import AlexNet

def draw_test_acc(epochs, val_accs):
    title="val accs"
    plt.title(title, fontsize=24)
    plt.xlabel("epoch", fontsize=14)
    plt.ylabel("acc", fontsize=14)
    epoch = list(range(epochs))
    plt.plot(epoch, val_accs, color='red', label='val accs')
    plt.legend()
    plt.grid()
    plt.show()


def main():
    # 判断CUDA是否可用，如果可用则使用GPU，否则使用CPU
    device = paddle.set_device('gpu' if paddle.is_compiled_with_cuda() else 'cpu')
    print("using {} device.".format(device))

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
    # 创建训练数据集对象，使用自定义的MyDataset类加载图像，并应用预定义的转换操作
    train_dataset = MyDataset(data_dir=os.path.join(image_path, "train"), transform=data_transform["train"])
    # 获取训练数据集的长度
    train_num = len(train_dataset)

    # 获取类别到索引的映射，并将其转换为索引到类别的字典：{'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_dict = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_dict.items())
    # 将类别索引字典转换为JSON格式字符串
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # 确定数据加载器的工作进程数
    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    # 创建训练数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

    # 创建验证数据集对象
    validate_dataset = MyDataset(data_dir=os.path.join(image_path, "val"), transform=data_transform["val"])
    # 获取验证数据集的长度
    val_num = len(validate_dataset)
    # 创建验证数据加载器
    validate_loader = DataLoader(validate_dataset, batch_size=4, shuffle=False, num_workers=nw)
    # 打印用于训练和验证的图像数量
    print("using {} images for training, {} images for validation.".format(train_num, val_num))
    # 定义一个AlexNet模型，num_classes参数指定了输出层的神经元数量，init_weights设置为True表示初始化网络权重
    net = AlexNet(num_classes=5)
    # 将模型移动到之前定义的设备上，如果使用GPU则移动到GPU，否则在CPU上运行
    # net.to(device)
    # 使用torchsummary库打印模型的结构信息，包括每层的输入输出大小和参数数量
    paddle.summary(net, input_size=(1, 3, 224, 224))  # input_size是一个元组，表示输入图像的通道数和空间尺寸
    # 定义交叉熵损失函数，用于分类任务
    loss_function = nn.CrossEntropyLoss()
    # 定义优化器，使用Adam算法，学习率设置为0.0002
    optimizer = optim.Adam(parameters=net.parameters(), learning_rate=0.0002)
    # 定义训练的轮数（epoch）
    epochs = 20
    # 定义模型保存路径
    save_path = './AlexNet.pdparams'
    # 定义最佳准确率变量，初始设置为0.0
    best_acc = 0.0
    # 记录每一轮在验证集上的准确率
    val_acc = []
    # 计算训练步数，即数据加载器中的样本总数除以batch_size
    train_steps = len(train_loader)

    for epoch in range(epochs):
        # 训练阶段
        net.train()
        # 初始化本轮训练的损失为0
        running_loss = 0.0
        # 使用tqdm库来显示进度条
        train_bar = tqdm(train_loader, file=sys.stdout)
        # 遍历训练数据加载器
        for step, data in enumerate(train_bar):
            images, labels = data  # 解包数据，获取图像和标签
            # labels = paddle.reshape(labels, (-1, 1))
            optimizer.clear_grad()  # 清空优化器的梯度，与PyTorch的优化器不同（zero_grad()）
            outputs = net(images)  # 将图像数据传入模型得到预测结果
            loss = loss_function(outputs, paddle.to_tensor(labels, dtype='int64'))  # 计算损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新网络权重

            # print statistics
            running_loss += loss.item()  # 累加损失值
            # 更新进度条描述，显示当前训练轮次和损失
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss.item())

        # 验证阶段
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch  初始化本轮验证的准确率累计
        # 在不计算梯度的情况下遍历验证数据加载器
        with paddle.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            # 遍历验证数据
            for val_data in val_bar:
                val_images, val_labels = val_data  # 解包验证数据
                # val_labels = paddle.reshape(val_labels, (-1, 1))
                outputs = net(val_images)  # 将图像数据传入模型得到预测结果
                predict_y = paddle.argmax(outputs, axis=1)  # 计算预测结果，与Pytorch不同
                # print('predict_y:',predict_y, '  val_labels', val_labels)
                # 计算模型在测试集上的精度
                acc += paddle.equal(predict_y, paddle.cast(val_labels,'int64')).sum().item()  # 累加正确预测的数量
        # 计算验证准确率并保存
        val_accurate = acc / val_num
        val_acc.append(val_accurate)
        # 打印每个epoch的训练损失和验证准确率
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        # 如果当前模型在验证集上的准确率是最佳，则保存模型
        if val_accurate > best_acc:
            best_acc = val_accurate
            paddle.save(net.state_dict(), save_path)
            print('Saved best model')

        # 绘制训练过程中的损失和准确率曲线
    draw_test_acc(epochs, val_acc)
    print("验证集最高准确率：", max(val_acc))

    print('Finished Training')

if __name__ == '__main__':
    main()

