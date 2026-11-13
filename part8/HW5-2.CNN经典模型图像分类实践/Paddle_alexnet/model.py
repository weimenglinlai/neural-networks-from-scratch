import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class AlexNet(nn.Layer):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2D(3, 48, kernel_size=11, stride=4, padding=2),  # input[1, 3, 224, 224] output[1, 48, 55, 55]
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2),                  # output[1, 48, 27, 27]
            nn.Conv2D(48, 128, kernel_size=5, padding=2),           # output[1, 128, 27, 27]
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2),                  # output[1, 128, 13, 13]
            nn.Conv2D(128, 192, kernel_size=3, padding=1),          # output[1, 192, 13, 13]
            nn.ReLU(),
            nn.Conv2D(192, 192, kernel_size=3, padding=1),          # output[1, 192, 13, 13]
            nn.ReLU(),
            nn.Conv2D(192, 128, kernel_size=3, padding=1),          # output[1, 128, 13, 13]
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2)                   # output[1, 128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = paddle.flatten(x, start_axis=1)
        x = self.classifier(x)
        return x

    # def _initialize_weights(self):
    #     for m in self.sublayers():
    #         if isinstance(m, nn.Conv2D):
    #             paddle.nn.initializer.KaimingNormal(out_dim=m.out_channels, act='relu')(m.weight)
    #             if m.bias:
    #                 paddle.nn.initializer.Constant(value=0)(m.bias)
    #         elif isinstance(m, nn.Linear):
    #             paddle.nn.initializer.Normal(loc=0., scale=0.01)(m.weight)
    #             paddle.nn.initializer.Constant(value=0)(m.bias)


if __name__ == '__main__':
    model = AlexNet(num_classes=5)
    # 用paddle.summary来查看模型的参数量。需要传入你的模型和输入数据的尺寸。
    params_info = paddle.summary(model, (1, 3, 224, 224))  # 参数依次为batchsize,channel,w,h
    print(params_info)