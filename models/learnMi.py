import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MIFCNet(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(MIFCNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_output = nn.Linear(latent_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc_output(x))
        return x

class Mine(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 1)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight, std=0.02)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output


class MIFCNetDIM(nn.Module):
    """Simple custom network for computing MI.

    """
    def __init__(self, n_input, n_units, bn=False):
        """
        Args:
            n_input: Number of input units.
            n_units: Number of output units.
        """
        super().__init__()

        self.bn = bn

        assert(n_units >= n_input)

        self.linear_shortcut = nn.Linear(n_input, n_units)
        self.block_nonlinear = nn.Sequential(
            nn.Linear(n_input, n_units, bias=False),
            nn.BatchNorm1d(n_units),
            nn.ReLU(),
            nn.Linear(n_units, n_units)
        )

        # initialize the initial projection to a sort of noisy copy
        eye_mask = np.zeros((n_units, n_input), dtype=np.uint8)
        for i in range(n_input):
            eye_mask[i, i] = 1

        self.linear_shortcut.weight.data.uniform_(-0.01, 0.01)
        self.linear_shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.)
        self.block_ln = nn.LayerNorm(n_units)

    def forward(self, x):
        h = self.block_nonlinear(x) + self.linear_shortcut(x)

        if self.bn:
            h = self.block_ln(h)
        return h


class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(16, 64, kernel_size=3, padding=1)  # 修改输入通道数为 16
        self.c1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.l0 = nn.Linear(32 * 1 * 96, 16)  # 修改这里以满足合适的输入输出大小
        self.l1 = nn.Linear(192 + 16, 208)
        self.l2 = nn.Linear(208, 1)

    def forward(self, y, M):
        h = F.relu(self.c0(M))  # 修改输入为 M
        h = self.c1(h)
        h = h.view(y.shape[0], -1)
        h = F.relu(self.l0(h))
        h = torch.cat((y, h), dim=1)
        h = F.relu(self.l1(h))
        return self.l2(h)


class LocalDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.c0 = nn.Conv2d(in_channels, 512, kernel_size=1)  # 修改输入通道数为 208
        self.c1 = nn.Conv2d(512, 512, kernel_size=1)
        self.c2 = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        return self.c2(h)


class DeepInfoMaxLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()

        self.alpha = alpha
        self.beta = beta

    def forward(self, y, M, M_prime):
        # 输入参数：
        # y: 编码器编码后的特征表示，shape: (128, 192)
        # M: 编码器提取的局部特征图，shape: (128, 16, 1, 96)
        # M_prime: 随机打乱的局部特征图，shape: (128, 16, 1, 96)

        # 对 y 进行维度扩展，以便与 M 和 M_prime 进行拼接
        y_exp = y.unsqueeze(-1).unsqueeze(-1)  # shape: (128, 192, 1, 1)
        y_exp = y_exp.expand(-1, -1, 1, M.shape[1])  # shape: (128, 192, 1, 96)

        # 将 y_exp 与 M 和 M_prime 进行拼接
        y_M = torch.cat((M, y_exp), dim=1)  # shape: (128, 16+192, 1, 96)
        y_M_prime = torch.cat((M_prime, y_exp), dim=1)  # shape: (128, 16+192, 1, 96)

        self.global_d = GlobalDiscriminator().to(device)
        self.local_d = LocalDiscriminator(y_M.shape[1]).to(device)
        # 计算局部损失
        Ej = -F.softplus(-self.local_d(y_M)).mean()  # 正样本的局部损失
        Em = F.softplus(self.local_d(y_M_prime)).mean()  # 负样本的局部损失
        LOCAL = (Em - Ej) * self.beta  # 总局部损失

        # 计算全局损失
        Ej = -F.softplus(-self.global_d(y, M)).mean()  # 正样本的全局损失
        Em = F.softplus(self.global_d(y, M_prime)).mean()  # 负样本的全局损失
        GLOBAL = (Em - Ej) * self.alpha  # 总全局损失

        # 返回总损失（局部损失 + 全局损失）
        return LOCAL + GLOBAL


def shuffle_tensor(tensor, patch_size=(4, 24)):
    """
    将给定的张量根据指定的 patch 大小沿 axis=3 随机打乱。

    参数:
        tensor (torch.Tensor): 输入的张量，形状为 (N, C, H, W)
        patch_size (tuple): 要划分的 patch 大小 (ph, pw)

    返回:
        shuffled_tensor (torch.Tensor): 打乱后的张量，形状与输入张量相同
    """

    # 将张量划分为 (ph, pw) 的 patches
    patches = tensor.unfold(3, patch_size[1], patch_size[1])

    # 获取打乱 patches 的随机索引
    perm = torch.randperm(patches.shape[2], device=tensor.device)

    # 用随机索引打乱 patches
    shuffled_patches = patches[:, :, perm, :]

    # 将打乱的 patches 重新组合成原始形状的张量
    shuffled_tensor = shuffled_patches.reshape(tensor.shape)

    return shuffled_tensor
