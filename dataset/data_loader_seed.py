import os
import numpy as np
import torch.utils.data as Data
import torch
from scipy.interpolate import interp2d
from sklearn import preprocessing
import scipy.io as scio


def get_dataset(file_name, type, resolution):
    data_list = []
    label_list = []
    domain = os.path.abspath('./data/SEED')
    data_path = os.path.join(domain, file_name)
    data = scio.loadmat(data_path)
    # 交叉熵label不能为负
    labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

    keys = list(data.keys())[3:]
    # 连续取值
    for key, label in zip(keys, labels):
        nums = data[key].shape[-1]
        i, j = 0, 200
        while j <= nums:
            array1 = data[key][:, i:j]

            # 生成原始矩阵的x和y坐标值
            x = np.arange(200)
            y = np.arange(62)
            # 创建插值函数
            interp_func = interp2d(x, y, array1)
            # 生成新矩阵的x和y坐标值
            new_x = np.linspace(0, 199, 384)
            new_y = np.linspace(0, 61, 62)
            # 计算新矩阵
            new_matrix = interp_func(new_x, new_y)

            # 绘制62*200原始数据和62*384插值后数据的热图
            # fig, axs = plt.subplots(1, 2, figsize=(20, 10))
            # fig.subplots_adjust(wspace=0.5)
            # im1 = axs[0].imshow(array1, cmap='viridis', aspect='auto')  # 图1：原始数据
            # axs[0].set_title('Original 62x200 Data')
            # im2 = axs[1].imshow(new_matrix, cmap='viridis', aspect='auto')  # 图2：插值后数据
            # axs[1].set_title('Interpolated 62x384 Data')
            # plt.show()

            data_list.append(new_matrix)
            label_list.append(label)
            i, j = j, j + 200
    X_data = np.reshape(data_list, (np.shape(data_list)[0], 1, 62, 384))
    X_label = np.reshape(label_list, (np.shape(data_list)[0], 1))
    if type == 'global_scale_value':
        X_data = (X_data - np.min(X_data)) / (np.max(X_data) - np.min(X_data))
    if type == "global_gaussian_value":
        X_data = (X_data - np.mean(X_data)) / np.std(X_data)
    if type == "pixel_scale_value":
        X_data = data_norm(X_data, resolution)
    if type == "origin":
        X_data = X_data
    # data_tmp = copy.deepcopy(X_data)
    # label_tmp = copy.deepcopy(X_label)
    # # 归一化
    # for i in range(len(data_tmp)):
    #     for j in range(len(data_tmp[0])):
    #         data_tmp[i][j] = utils.norminy(data_tmp[i][j])
    return X_data, X_label


# # 假设batch_size为64，每条数据维度为192
# mi = MIFCNet(384, 192)
# batch_size = 64
# data1 = torch.randn(batch_size, 192)
# data2 = torch.randn(batch_size, 192)
# optimizer = optim.Adam(mi.parameters(), lr=0.001)
def data_norm(X_data, resolution):
    X_data_sque = np.reshape(X_data, (np.shape(X_data)[0], 62 * resolution))
    X_data_sque_scaled = max_min_scale(X_data_sque)
    X_data = np.reshape(X_data_sque_scaled, (np.shape(X_data)[0], 1, 62, resolution))
    return X_data


def max_min_scale(data_sque):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    data_sque = min_max_scaler.fit_transform(data_sque)
    return data_sque

def get_TensorDataset_seed(sub):
    BATCH_SIZE = 128
    path = "./data/SEED"
    path_list = os.listdir(path)
    train_dataTensor = torch.empty(0, 1, 62, 384)
    train_labelsTensor = torch.empty(0, 1)
    valid_dataTensor = torch.empty(0, 1, 62, 384)
    valid_labelsTensor = torch.empty(0, 1)
    total_features = []
    total_labels = []
    print(len(path_list))
    for i in range(0, len(path_list), 3):
        batch_files = path_list[i]
        # X_data, X_label = get_dataset(batch_files,"global_scale_value",384)
        X_data, X_label = get_dataset(batch_files, type, 384)
        total_labels.append(X_label)
        X_data = X_data.astype('float32')
        X_data = torch.from_numpy(X_data)
        num = batch_files[:2] if batch_files[:2].isdigit() else batch_files[0]
        if int(num) == (sub + 1):
            valid_dataTensor = torch.cat((valid_dataTensor, X_data), dim=0)
            labels = torch.Tensor(X_label)
            valid_labelsTensor = torch.cat((valid_labelsTensor, labels), dim=0)
        else:
            train_dataTensor = torch.cat((train_dataTensor, X_data), dim=0)
            labels = torch.Tensor(X_label)
            train_labelsTensor = torch.cat((train_labelsTensor, labels), dim=0)
    torch_dataset_train = Data.TensorDataset(train_dataTensor, train_labelsTensor)
    torch_dataset_valid = Data.TensorDataset(valid_dataTensor, valid_labelsTensor)
    loader_train = Data.DataLoader(
        dataset=torch_dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    loader_valid = Data.DataLoader(
        dataset=torch_dataset_valid,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    return loader_train, loader_valid