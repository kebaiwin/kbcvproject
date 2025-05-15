import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义了第一个二维卷积层
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # 定义了第二个二维卷积层
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # 定义了第一个全连接层
        self.fc1 = nn.Linear(in_features=64 * 5 * 5, out_features=128)
        # 定义了第二个全连接层
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    # 定义前向传播函数
    def forward(self, x):
        # 对输入数据 x 应用第一个卷积层 conv1，然后使用 ReLU（Rectified Linear Unit）激活函数
        x = nn.functional.relu(self.conv1(x))
        # 对经过 conv1 和 ReLU 激活后的特征图应用最大池化操作
        x = nn.functional.max_pool2d(x, kernel_size=2)
        # 对输入数据 x 应用第二个卷积层 conv2，然后使用 ReLU（Rectified Linear Unit）激活函数
        x = nn.functional.relu(self.conv2(x))
        # 对经过 conv2 和 ReLU 激活后的特征图应用最大池化操作
        x = nn.functional.max_pool2d(x, kernel_size=2)
        # 将经过两次卷积和两次池化后的特征图进行形状变换，以便输入到全连接层
        x = x.view(-1, 64 * 5 * 5)
        # 将展平后的向量输入到第一个全连接层 fc1，然后使用 ReLU 激活函数。
        x = nn.functional.relu(self.fc1(x))
        # 将经过 fc1 和 ReLU 激活后的向量输入到第二个全连接层 fc2，得到最终的输出，这个输出通常会用于后续的分类任务
        x = self.fc2(x)
        return x

model = NeuralNetwork()
print(model)