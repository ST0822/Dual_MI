from datetime import datetime
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import DataLoader
import mne
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import random

from torch.utils.tensorboard import SummaryWriter


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_data():
    # 文件夹路径，包含MAT文件
    folder_path = '../data/2_preprocessed_byJANE'
    # 初始化一个空的列表来存储所有矩阵
    all_matrices = []
    # 遍历文件夹中的所有MAT文件
    label_file = [f for f in os.listdir(folder_path) if 'deap_feature' in f][0]
    label_file = os.path.join(folder_path, label_file)
    label = np.array(loadmat(label_file)['label'])
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.startswith('s') and file_name.endswith('.mat'):
            file_path = os.path.join(folder_path, file_name)
            # 加载MAT文件
            mat_data = loadmat(file_path)
            # 假设clean_data是一个结构体数组，每行是一个结构体
            # 如果clean_data实际上是一个结构体而不是结构体数组，则需要调整下面的代码
            if isinstance(mat_data['clean_data'], np.ndarray) and mat_data['clean_data'].dtype.names:
                # 遍历结构体数组中的每一行
                for row in mat_data['clean_data']:
                    # 提取'Geneva_video'字段
                    geneva_video = row['Geneva_video']

                    # 将矩阵添加到列表中
                    all_matrices.append(geneva_video)
            else:
                # 如果clean_data不是预期的结构体数组，则可能需要不同的处理方式
                print(f"Warning: Unexpected data structure in {file_name}. Skipping.")
            # 将列表转换为NumPy数组（可选，取决于您是否需要这样做）
    # 注意：这将创建一个形状为(1280, 32, 30720)的数组，其中1280是矩阵的数量
    all_matrices_array = np.array(all_matrices).reshape(1280)
    all_matrices_array = np.concatenate(all_matrices_array, axis=1)
    return all_matrices_array, label


def pre_process3(data, sfreq=512):
    ch_names = ['eeg_' + str(i + 1) for i in range(32)]
    info = mne.create_info(
        # 通道名
        ch_names=ch_names,
        # 通道类型
        ch_types=['eeg' for _ in range(32)],
        # 采样频率
        sfreq=sfreq
    )
    # raw_data = torch.tensor([])
    raw_data = torch.empty(0, 1, 32, 384)
    data = data[:, :((np.shape(data)[1]) // sfreq) * sfreq]
    for i in range(5):
        sec_data = data[:, 256*i*30720:256*(i+1)*30720]
        raw = mne.io.RawArray(sec_data, info)
        # 重采样
        raw = raw.copy().resample(sfreq=384)
        sec_data, _ = raw[:]
        # print(data[i+2, j+1][:, 0])
        # 标准化
        # data[i+2, j+1] = norm_data(data[i+2, j+1], norm_type='linear')
        sec_data = preprocessing.scale(sec_data)
        # print(data[i+2, j+1][:, 0])
        # reshape成（秒，1，通道，每秒）便于DataLoader
        sec_data = sec_data.reshape(32, 1, np.shape(sec_data)[1] // 384, 384)
        sec_data = np.swapaxes(sec_data, 0, 2)
        # 转为张量数据
        sec_data = torch.tensor(sec_data, dtype=torch.float32)
        # 矩阵拼接
        # raw_data = np.append(np.array([]), data[i+2, j+1], axis=0)
        raw_data = torch.cat([raw_data, sec_data], dim=0)
    return raw_data



class EEGfuseNet_Channel_32(nn.Module):
    def __init__(self, hidden_dim, n_layer, n_filters, input_size):
        super(EEGfuseNet_Channel_32, self).__init__()
        ## Channel 32 network for DEAP,HCI dataset.
        # conventional convolution for time dim
        self.conv1 = nn.Conv2d(1, int(16 * n_filters), (1, int(input_size / 2 + 1)), stride=1,
                               padding=(0, int(input_size / 4)))
        self.batchNorm1 = nn.BatchNorm2d(int(16 * n_filters), False)
        self.n_filters = n_filters
        self.hidden_dim = hidden_dim
        self.length = input_size / 32
        # spatial convolution for channel dim
        self.depthwiseconv2 = nn.Conv2d(int(16 * n_filters), int(32 * n_filters), (32, 1),
                                        padding=0)  ## 32 channel EEG signal from DEAP,HCI dataset, so the kernel length here is 32
        self.batchNorm2 = nn.BatchNorm2d(int(32 * n_filters), False)
        self.pooling1 = nn.MaxPool2d((1, 4), return_indices=True)

        # depthwise separable  convolutions
        self.separa1conv3 = nn.Conv2d(int(32 * n_filters), int(32 * n_filters), (1, int(input_size / 8 + 1)), stride=1,
                                      padding=(0, int(input_size / 16)), groups=int(32 * n_filters))  # (32, 1, T/5)
        self.separa2conv4 = nn.Conv2d(int(32 * n_filters), int(16 * n_filters), 1)
        self.batchNorm3 = nn.BatchNorm2d(int(16 * n_filters), False)
        self.pooling2 = nn.MaxPool2d((1, 8), return_indices=True)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.25)
        self.dropout3 = nn.Dropout(p=0.25)
        self.dropout4 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(int(16 * n_filters), int(16 * n_filters))
        self.fc2 = nn.Linear(int(hidden_dim * 2 * n_filters), int(hidden_dim * n_filters))
        self.fc3 = nn.Linear(int(hidden_dim * n_filters), int(hidden_dim * 2 * n_filters))
        self.fc4 = nn.Linear(int(2 * 16 * n_filters), int(16 * n_filters))
        # GRU
        self.gru_en = nn.GRU(int(16 * n_filters), int(hidden_dim * n_filters), n_layer, batch_first=True,
                             bidirectional=True)
        self.gru_de = nn.GRU(int(2 * hidden_dim * n_filters), int(16 * n_filters), n_layer, batch_first=True,
                             bidirectional=True)
        self.lstm = nn.LSTM(int(16 * n_filters), int(hidden_dim * n_filters), n_layer, batch_first=True,
                            bidirectional=True)
        # deconventional
        self.unpooling2 = nn.MaxUnpool2d((1, 8))
        self.batchnorm4 = nn.BatchNorm2d(int(32 * n_filters), False)
        self.desepara2conv4 = nn.ConvTranspose2d(int(16 * n_filters), int(32 * n_filters), 1)
        self.desepara1conv3 = nn.ConvTranspose2d(int(32 * n_filters), int(32 * n_filters), (1, int(input_size / 8 + 1)),
                                                 stride=1, padding=(0, int(input_size / 16)),
                                                 groups=int(32 * n_filters))

        # de spatial convolution for channel dim
        self.unpooling1 = nn.MaxUnpool2d((1, 4))
        self.batchnorm5 = nn.BatchNorm2d(int(16 * n_filters), False)  #
        self.dedepthsepara1conv3 = nn.ConvTranspose2d(int(32 * n_filters), int(16 * n_filters), (32, 1), stride=1,
                                                      padding=0)

        # de spatial convolution for channel dim
        self.deconv1 = nn.ConvTranspose2d(int(16 * n_filters), 1, (1, int(input_size / 2 + 1)), stride=1,
                                          padding=(0, int(input_size / 4)))

    def forward(self, x):
        # encoder
        self.index_pool = []
        self.output_pool = []
        x = self.conv1(x)  # (128,16,32,384)
        x = self.batchNorm1(x)  # (128,16,32,384)
        # Layer 2
        x = self.depthwiseconv2(x)  # (128,32,1,384)
        self.output_pool.append(x)
        x = self.batchNorm2(x)  # (128,32,1,384)
        self.output_pool.append(x)
        x = F.elu(x)  # (128,32,1,384)
        self.output_pool.append(x)
        x, idx2 = self.pooling1(x)  # (128,32,1,96)get data1 and their index after pooling
        self.output_pool.append(x)
        self.index_pool.append(idx2)
        x = self.dropout1(x)  # (128,32,1,96)
        self.output_pool.append(x)
        # Layer 3
        x = self.separa1conv3(x)  # (128,32,1,96)
        self.output_pool.append(x)
        x = self.separa2conv4(x)  # (128,16,1,96)
        self.output_pool.append(x)
        x = self.batchNorm3(x)  # (128,16,1,96)
        self.output_pool.append(x)
        x = F.elu(x)  # (128,16,1,96)
        self.output_pool.append(x)
        feature_map, idx3 = self.pooling2(x)  # (128,16,1,12)
        self.output_pool.append(feature_map)

        self.index_pool.append(idx3)
        # Layer 4：FC Layer
        x = feature_map.permute(0, 3, 2, 1)  # (128,12,1,16)
        self.output_pool.append(x)
        x = x[:, :, -1, :, ]  # (128,12,16)

        x = self.fc1(x)  # (128,12,16)

        x = F.elu(x)  # (128,12,16)

        out, _ = self.gru_en(x)  # (128,12,32)

        x = out  # (128,12,32)
        x = self.fc2(x)  # (128,12,16)

        code = x.reshape(x.shape[0], int(16 * self.n_filters) * int(self.length))  # (128,192)

        feature_map = feature_map.reshape(
            (feature_map.shape[0], int(16 * self.n_filters) * int(self.length)))  # (128,192)

        return code, feature_map, self.output_pool

    def decoder(self, code):
        x = code.reshape(code.shape[0], int(self.length), int(16 * self.n_filters))
        x = self.fc3(x)
        out, _ = self.gru_de(x)
        x = out
        x = self.fc4(x)
        x = F.elu(x)
        x = x.reshape((x.shape[0], x.shape[1], 1, x.shape[2]))
        x = x.permute(0, 3, 2, 1)
        x = self.unpooling2(x, self.index_pool[1])

        x = self.desepara2conv4(x)
        x = self.desepara1conv3(x)
        x = self.batchnorm4(x)
        x = self.dropout4(x)

        # Layer 3
        x = F.elu(x)
        x = self.unpooling1(x, self.index_pool[0])
        x = self.dedepthsepara1conv3(x)
        x = self.batchnorm5(x)
        # Layer 4
        x = self.deconv1(x)
        return x

class Discriminator_Channel_32(nn.Module):
    def __init__(self, n_layer, n_filters, input_size):
        super(Discriminator_Channel_32, self).__init__()
        self.conv1 = nn.Conv2d(1, int(8 * n_filters), (1, int(input_size / 2 + 1)), stride=1,
                               padding=(0, int(input_size / 4)))  # (16, C, T)
        self.batchNorm1 = nn.BatchNorm2d(8 * n_filters, False)
        self.length = input_size / 32
        # Layer 2：spatial convolution for channel dim
        self.depthwiseconv2 = nn.Conv2d(int(8 * n_filters), int(16 * n_filters), (32, 1), padding=0)
        self.batchNorm2 = nn.BatchNorm2d(int(16 * n_filters), False)
        self.pooling1 = nn.MaxPool2d((1, 4), return_indices=False)
        self.separa1conv3 = nn.Conv2d(int(16 * n_filters), int(16 * n_filters), (1, int(input_size / 8 + 1)), stride=1,
                                      padding=(0, int(input_size / 16)), groups=int(16 * n_filters))
        self.separa2conv4 = nn.Conv2d(int(16 * n_filters), int(8 * n_filters), 1)
        self.batchNorm3 = nn.BatchNorm2d(int(8 * n_filters), False)
        self.pooling2 = nn.MaxPool2d((1, 8), return_indices=False)
        self.fc1 = nn.Linear(int(self.length * 8), 1)

    def forward(self, x):
        # encoder
        x = self.conv1(x)
        x = self.batchNorm1(x)
        # Layer 2
        x = self.depthwiseconv2(x)
        x = self.batchNorm2(x)
        x = F.elu(x)
        x = self.pooling1(x)  # get data and their index after pooling
        # Layer 3
        x = self.separa1conv3(x)
        x = self.separa2conv4(x)
        x = self.batchNorm3(x)
        x = F.elu(x)
        x = self.pooling2(x)

        # Layer 4：FC Layer
        x = x.reshape((x.shape[0], int(self.length * 8)))
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x


class GlobalDiscriminator(nn.Module):
    def __init__(self, c0_in, l0_in, l1_in):
        super().__init__()
        self.c0 = nn.Conv2d(c0_in, 64, kernel_size=3, padding=1)  # 修改输入通道数为 16
        self.c1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.l0 = nn.Linear(l0_in, 128)  # 修改这里以满足合适的输入输出大小
        self.l1 = nn.Linear(l1_in, l1_in)
        self.l2 = nn.Linear(l1_in, 1)

    def forward(self, y, M):
        h = F.relu(self.c0(M))  # 修改输入为 M
        h = self.c1(h)
        h = h.view(y.shape[0], -1)
        h = F.relu(self.l0(h))
        h = torch.cat((y, h), dim=1)
        h = F.relu(self.l1(h))
        return self.l2(h)


# class LocalDiscriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.c0 = nn.Conv2d(208, 512, kernel_size=1)  # 修改输入通道数为 208
#         self.c1 = nn.Conv2d(512, 512, kernel_size=1)
#         self.c2 = nn.Conv2d(512, 1, kernel_size=1)
#
#     def forward(self, x):
#         h = F.relu(self.c0(x))
#         h = F.relu(self.c1(h))
#         return self.c2(h)

class LinearSimilarityDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # 定义多个线性层
        self.fc1 = nn.Linear(input_dim, 128)  # 首先将288维映射到128维
        self.fc2 = nn.Linear(128, 64)  # 将128维进一步映射到64维
        self.fc3 = nn.Linear(64, 1)  # 最后将64维映射到1维

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 由于输入形状是(128, 1, 16, 288)，我们需要先调整一下形状
        shape = x.shape[2]
        x = x.view(-1, x.shape[3])  # 将输入展平为(128*16, 288)

        # 通过多个线性层
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # 输出形状将会是(128*16, 1)

        # 复原回(128, 1, 16, 1)的形状
        x = x.view(-1, 1, shape, 1)  # 将形状调整回(128, 1, 16, 1)
        return x
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
        # M: 编码器提取的局部特征图，shape: (128, 1, 16, 96)
        # M_prime: 随机打乱的局部特征图，shape: (128, 1, 16, 96)
        self.global_d = GlobalDiscriminator(M.shape[1], 32 * (M.shape[3]), 128 + y.shape[1]).to(device)
        # 计算全局损失
        Ej = -F.softplus(-self.global_d(y, M)).mean()  # 正样本的全局损失
        Em = F.softplus(self.global_d(y, M_prime)).mean()  # 负样本的全局损失
        GLOBAL = (Em - Ej) * self.alpha  # 总全局损失

        M = M.permute(0, 2, 1, 3)
        M_prime = M_prime.permute(0, 2, 1, 3)
        # 对 y 进行维度扩展，以便与 M 和 M_prime 进行拼接
        y_exp = y.unsqueeze(1).unsqueeze(2)
        y_exp = y_exp.expand(-1, -1,  M.shape[2], -1)
        # 将 y_exp 与 M 和 M_prime 进行拼接
        y_M = torch.cat((M, y_exp), dim=3)  # shape: (128, 1, 16, 96+192)
        y_M_prime = torch.cat((M_prime, y_exp), dim=3)  # shape: (128, 1, 16, 96+192)
        self.local_d = LinearSimilarityDiscriminator(y_M.shape[3]).to(device)
        # 计算局部损失
        Ej = -F.softplus(-self.local_d(y_M)).mean()  # 正样本的局部损失
        Em = F.softplus(self.local_d(y_M_prime)).mean()  # 负样本的局部损失
        LOCAL = (Em - Ej) * self.beta  # 总局部损失
        # 返回总损失（局部损失 + 全局损失）
        return LOCAL + GLOBAL


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





def shuffle_tensor(tensor, patch_size=(4, 24)):
    """
    将给定的张量根据指定的 patch 大小沿 axis=3 随机打乱。

    参数:
        tensor (torch.Tensor): 输入的张量，形状为 (N, C, H, W)
        patch_size (tuple): 要划分的 patch 大小 (ph, pw)

    返回:
        shuffled_tensor (torch.Tensor): 打乱后的张量，形状与输入张量相同
    """
    shuffled_tensor = tensor[:, torch.randperm(tensor.shape[1]), :, :]

    # 将张量划分为 (ph, pw) 的 patches
    # patches = tensor.unfold(3, patch_size[1], patch_size[1])
    #
    # # 获取打乱 patches 的随机索引
    # perm = torch.randperm(patches.shape[2], device=tensor.device)
    #
    # # 用随机索引打乱 patches
    # shuffled_patches = patches[:, :, perm, :]
    #
    # # 将打乱的 patches 重新组合成原始形状的张量
    # shuffled_tensor = shuffled_patches.reshape(tensor.shape)

    return tensor, shuffled_tensor


def pretrain(load_pretrained=False):
    """
    Initialize and optionally load pre-trained generator and discriminator models.

    Args:
        load_pretrained (bool): If True, load pre-trained model. Defaults to False.

    Returns:
        generator (nn.Module): The generator model.
        discriminator (nn.Module): The discriminator model.
        optimizer_G (torch.optim.Adam): The optimizer for the generator.
        optimizer_D (torch.optim.Adam): The optimizer for the discriminator.
    """
    # 实例化
    generator = EEGfuseNet_Channel_32(16, 1, 1, 384).to(device).float()
    discriminator = Discriminator_Channel_32(1, 1, 384).to(device).float()
    infomax_loss = DeepInfoMaxLoss().to(device).float()

    file_path = __file__
    file_name = os.path.basename(file_path)
    log_dir = os.path.join('../logs', file_name, f'{sub}_MIlog')
    # with SummaryWriter(log_dir=log_dir,
    #                    comment='train') as writer:  # 可以直接使用python的with语法，自动调用close方法
    #     # TensorBoard
    #     input = torch.zeros((128, 1, 62, 384)).to(device)
    #     writer.add_graph(generator, input)
    #     writer.add_graph(discriminator, input)

    # 模型初始化
    generator.apply(weigth_init)
    discriminator.apply(weigth_init)
    infomax_loss.apply(weigth_init)

    # 预训练
    if load_pretrained:
        os.chdir('../data')
        generator.load_state_dict(torch.load('Pretrained_model_SEED.pkl'))

    # 优化器
    # optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=0.001)
    # optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=0.0002)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.001)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    optimizer_infomax = torch.optim.Adam(generator.parameters(), lr=0.001)

    return generator, discriminator, infomax_loss, optimizer_G, optimizer_D, optimizer_infomax


def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.3)

    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        m.bias.data.zero_()


def my_dataset(raw_data, train_index, test_index, batch_size=128):
    # Convert raw_data to single-precision float
    # raw_data = torch.tensor(raw_data, dtype=torch.float32)

    # Create train and test subsets
    train_fold = torch.utils.data.dataset.Subset(raw_data, train_index)
    test_fold = torch.utils.data.dataset.Subset(raw_data, test_index)

    # Package into DataLoader for training
    train_loader = DataLoader(dataset=train_fold, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(dataset=test_fold, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, test_loader


def train_epoch(train_loader, generator, discriminator, optimizer_G, optimizer_D, device, num):
    g_loss = 0.0
    d_loss = 0.0

    generator.train()
    discriminator.train()
    infomax_loss.train()

    for i, train_data in enumerate(train_loader):
        train_data = train_data.float().to(device)

        true_label = torch.ones((train_data.shape[0], 1)).to(device)
        false_label = torch.zeros((train_data.shape[0], 1)).to(device)

        # Train Generator
        optimizer_G.zero_grad()
        optimizer_infomax.zero_grad()
        latent_code, feature_map1, feature_map_list = generator(train_data)
        # x, feature, z = generator(train_data)
        recon = generator.decoder(latent_code)
        feature_map, shuffle_feature_map = shuffle_tensor(feature_map_list[num])
        G_Loss = autoencoder_loss(train_data, recon) + infomax_loss(
            latent_code, feature_map, shuffle_feature_map)
        g_loss += G_Loss.item()

        G_Loss.backward()
        optimizer_G.step()
        optimizer_infomax.step()

        # Train Discriminator
        optimizer_D.zero_grad()

        D_Loss = 0.1 * (discriminator_loss(discriminator(train_data), true_label) + discriminator_loss(
            discriminator(recon.detach()), false_label))
        d_loss += D_Loss.item()

        D_Loss.backward()
        optimizer_D.step()

    return g_loss / len(train_loader), d_loss / len(train_loader)


def valid_epoch(test_loader, generator, discriminator, device, num):
    features = np.zeros((0, 192))
    test_loss = 0.0

    generator.eval()
    discriminator.eval()
    infomax_loss.eval()

    for j, test_data in enumerate(test_loader):
        test_data = test_data.float().to(device)

        true_label = torch.ones((test_data.shape[0], 1)).to(device)
        false_label = torch.zeros((test_data.shape[0], 1)).to(device)

        with torch.no_grad():
            latent_code, feature_map1, feature_map_list = generator(test_data)
            # x, feature, z = generator(test_data)
            recon = generator.decoder(latent_code)
            feature_map, shuffle_feature_map = shuffle_tensor(feature_map_list[num])

            test_Loss = autoencoder_loss(test_data, recon) + infomax_loss(latent_code, feature_map, shuffle_feature_map)
            test_loss += test_Loss.item()

            features = np.vstack((features, latent_code.cpu().detach().numpy()))

    return test_loss / len(test_loader), features


def knn_cross_validation(features, labels, n_splits=32):
    # KNN实例化
    knn = KNeighborsClassifier(n_neighbors=5)

    # KNN交叉验证
    kf = KFold(n_splits=n_splits, shuffle=False, random_state=None)

    score_sum = 0.0
    for train_index, test_index in kf.split(features):
        # 获取训练和测试数据
        train_data = features[train_index]
        train_label = labels[train_index]
        test_data = features[test_index]
        test_label = labels[test_index]

        # 训练KNN模型并预测
        knn.fit(train_data, train_label.ravel())
        score = knn.score(test_data, test_label)

        # 累加得分
        score_sum += score
        print("Fold score:", score)

    # 计算平均得分
    average_score = score_sum / n_splits
    return average_score


if __name__ == '__main__':
    # 代码
    # 模型加载选择GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.device_count())

    # 在训练脚本开始时设置seed
    seed = 42
    set_seed(seed)
    data, label = get_data()
    raw_data = pre_process3(data, sfreq=512)
    # Loss
    autoencoder_loss = nn.MSELoss(reduction='mean').to(device)
    generator_loss = nn.BCELoss().to(device)
    discriminator_loss = nn.BCELoss().to(device)

    # 交叉验证
    kf = KFold(n_splits=32, shuffle=False, random_state=None)
    feature_list = np.zeros((0, 192))
    epochs = 100

    sub = 0
    for train_index, test_index in kf.split(raw_data):
        sub += 1
        for num in range(11):
            generator, discriminator, infomax_loss, optimizer_G, optimizer_D, optimizer_infomax = pretrain(
                load_pretrained=False)
            train_loader, test_loader = my_dataset(raw_data, train_index, test_index, batch_size=128)

            # for epoch in range(1):
            for epoch in range(epochs):
                g_loss, d_loss = train_epoch(train_loader, generator, discriminator, optimizer_G, optimizer_D, device, num)
                test_loss, z = valid_epoch(test_loader, generator, discriminator, device, num)

                # Print training information
                print(f"[Sub {sub}] [Epoch {epoch + 1}/{epochs}] [G loss: {g_loss:.6f}] [D loss: {d_loss:.6f}]")

                # Print testing information
                print(f"[Sub {sub}] [Epoch {epoch + 1}/{epochs}] [Test loss: {test_loss:.6f}]")

                file_path = __file__
                file_name = os.path.basename(file_path)
                log_dir = os.path.join('../logs', file_name, f'{sub}_MIlog')
                with SummaryWriter(log_dir=log_dir,
                                   comment='train') as writer:  # 可以直接使用python的with语法，自动调用close方法
                    # 损失函数图像
                    writer.add_scalar('train_loss', g_loss, epoch + 1)
                    writer.add_scalar('dis_loss', d_loss, epoch + 1)
                    writer.add_scalar('test_loss', test_loss, epoch + 1)

            # feature_list = np.append(feature_list, z, axis=0)
            print("feature_list:" + str(np.shape(feature_list)))

            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = os.path.join('../checkpoints', file_name,
                                f'MSE+Dis_sub{sub}_featuremap{num}_{now}.pkl')
            torch.save(generator.state_dict(), path)

    # average_score = knn_cross_validation(feature_list, label)
    # print("Average score:", average_score)