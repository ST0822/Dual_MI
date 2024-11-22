import torch
import torch.nn as nn


class EnhancedSimilarityDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # 第一层卷积
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 3), padding=(0, 1))  # 输出形状: (128, 16, 16, 288)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()

        # 第二层卷积
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 3), padding=(0, 1))  # 输出形状: (128, 32, 16, 288)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        # 池化层
        self.pool = nn.MaxPool2d((1, 2))  # 降维: 输出形状将变为 (128, 32, 16, 144)

        # 最后一层卷积以得到相似度
        self.conv3 = nn.Conv2d(32, 1, kernel_size=(1, 144))  # 输出形状: (128, 1, 16, 1)

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h)

        h = self.pool(h)

        h = self.conv3(h)
        return h  # 输出形状 (128, 1, 16, 1)


class LinearSimilarityDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # 定义多个线性层
        self.fc1 = nn.Linear(288, 128)  # 首先将288维映射到128维
        self.fc2 = nn.Linear(128, 64)  # 将128维进一步映射到64维
        self.fc3 = nn.Linear(64, 1)  # 最后将64维映射到1维

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 由于输入形状是(128, 1, 16, 288)，我们需要先调整一下形状
        x = x.view(-1, 288)  # 将输入展平为(128*16, 288)

        # 通过多个线性层
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # 输出形状将会是(128*16, 1)

        # 复原回(128, 1, 16, 1)的形状
        x = x.view(-1, 1, 16, 1)  # 将形状调整回(128, 1, 16, 1)

        return x
# 测试代码
if __name__ == "__main__":
    x = torch.randn(128, 1, 16, 288)

    model = LinearSimilarityDiscriminator()
    output = model(x)

    print(output.shape)