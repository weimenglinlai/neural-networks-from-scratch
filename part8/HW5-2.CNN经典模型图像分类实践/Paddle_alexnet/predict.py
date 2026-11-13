import os
import json
import paddle
from PIL import Image
from paddle.vision import transforms
import paddle.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from model import AlexNet  # 确保你有一个与PaddlePaddle兼容的AlexNet模型定义

def main():
    # 判断CUDA是否可用，如果可用则使用GPU，否则使用CPU
    device = paddle.set_device('gpu' if paddle.is_compiled_with_cuda() else 'cpu')
    print("using {} device.".format(device))

    # 数据预处理转换操作
    data_transform = transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)  调整图像大小为224x224
                                   transforms.ToTensor(),  # 转换为Tensor
                                   transforms.Normalize(mean = [127.5, 127.5, 127.5], std = [127.5, 127.5, 127.5])])

    # 加载图像
    img_path = "./rose.jpg"
    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
    img = Image.open(img_path)
    img = img.convert('RGB')  # 确保图像是 RGB 格式
    img = np.array(img)

    # 可视化图像
    plt.imshow(img)

    # 飞桨训练时内部数据格式默认为float32，将图像数据格式转换为 float32（***非常重要，否则预测结果不对***）
    img = img.astype('float32')

    # 应用预处理转换
    img = data_transform(img)
    # 增加批次维度，增加后img的维度依次为batchsize,channel,w,h
    img = paddle.unsqueeze(img, axis=0)

    # 读取类别索引
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)
    print('class_indict: ', class_indict)

    # 创建模型并打印模型信息
    model = AlexNet(num_classes=5)
    paddle.summary(model, (1, 3, 224, 224))  # 参数依次为batchsize,channel,w,h

    # 加载模型权重
    weights_path = "./AlexNet.pdparams"
    assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
    model.set_state_dict(paddle.load(weights_path))

    # 设置为模型推断模式（不计算梯度，不更新模型参数）
    model.eval()

    with paddle.no_grad():     # 不计算梯度
        # 预测类别
        output = paddle.squeeze(model(img))
        # 获取概率分布
        predict = F.softmax(output, axis=0)
        # 获取最可能的类别索引
        predict_cla = paddle.argmax(predict).item()
    # 打印预测结果
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].item())   # 使用 .item() 来获取 Python 浮点数
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].item()))   # 使用 .item() 来获取 Python 浮点数

    # 可视化预测结果
    plt.show()

if __name__ == '__main__':
    main()