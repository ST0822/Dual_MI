import os
import time
import warnings

import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
import scipy.io as scio
import numpy as np
from numpy import shape
from torch.autograd import Variable
# from cuml.neighbors import KNeighborsClassifier
import torch.nn.functional as F
import torch.utils.data as Data

from models.Generator import EEGfuseNet_SEED


def get_norm_type(filename, norm_types):
    norm_types_set = set(norm_types)
    parts = filename.split('seed_')
    for part in parts:
        for norm_type in norm_types_set:
            if part.startswith(norm_type):
                return norm_type


def get_dataset(file_name, norm_type, resolution, path):
    data_list = []
    label_list = []
    domain = os.path.abspath(path)
    data_path = os.path.join(domain, file_name)
    data = scio.loadmat(data_path)
    # 交叉熵label不能为负
    labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

    keys = list(data.keys())[3:]
    # print(len(keys))
    # print(data[keys[0]].shape)
    # print(data[keys[0]].shape[-1])
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
    if norm_type == 'global_scale_value':
        X_data = (X_data - np.min(X_data)) / (np.max(X_data) - np.min(X_data))
    if norm_type == "global_gaussian_value":
        X_data = (X_data - np.mean(X_data)) / np.std(X_data)
    if norm_type == "pixel_scale_value":
        X_data = data_norm(X_data, resolution)
    if norm_type == "origin":
        X_data = X_data
    # data_tmp = copy.deepcopy(X_data)
    # label_tmp = copy.deepcopy(X_label)
    # # 归一化
    # for i in range(len(data_tmp)):
    #     for j in range(len(data_tmp[0])):
    #         data_tmp[i][j] = utils.norminy(data_tmp[i][j])
    return X_data, X_label


def data_norm(X_data, resolution):
    X_data_sque = np.reshape(X_data, (np.shape(X_data)[0], 62 * resolution))
    X_data_sque_scaled = max_min_scale(X_data_sque)
    X_data = np.reshape(X_data_sque_scaled, (np.shape(X_data)[0], 1, 62, resolution))
    return X_data


def max_min_scale(data_sque):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    data_sque = min_max_scaler.fit_transform(data_sque)
    return data_sque


def extractor_cnn_rnn_para(loader_train, feature_size, label_dim, filename, model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EEGfuseNet_SEED(16, 1, 1, 384).to(device)
    model.load_state_dict(torch.load(model_path + '/' + filename, map_location='cpu'))
    model.eval()
    code_mat = np.zeros((1, feature_size))
    label_mat = np.zeros((1, label_dim))
    with torch.no_grad():
        for step, (data, label) in enumerate(loader_train):
            inputs, label = Variable(data.to(device)), Variable(label.to(device))
            code, _ = model(inputs)
            code = code.cpu().detach().numpy()
            code = code.reshape((code.shape[0], feature_size))
            label = label.cpu().detach().numpy()
            label = label.reshape((label.shape[0], label_dim))
            code_mat = np.row_stack((code_mat, code))
            label_mat = np.row_stack((label_mat, label))
    total_features_2d = code_mat[1:len(code_mat), :]
    total_labels_2d = label_mat[1:len(label_mat), :]
    total_labels_1d = total_labels_2d.flatten()
    return total_features_2d, total_labels_1d


def EEGknn(total_feature, total_label, labels, neighbors):
    kk = len(labels)  # 留一交叉验证
    index1, index2 = 0, int(len(labels[0]))
    test_acc = 0
    test_accArray = []
    test_accArray = np.array(test_accArray)
    for k in range(kk):
        step = int(len(labels[k]))
        # 设置训练和测试集区间
        q=total_feature[:index1]
        r=total_feature[index2:]
        train_features = np.append(total_feature[:index1], total_feature[index2:],axis = 0)
        train_label = np.append(total_label[:index1], total_label[index2:],axis = 0)
        #
        test_features = (total_feature[index1:index2])
        test_label = (total_label[index1:index2])
        index1 = index2
        index2 = index2 + step
        clf = KNeighborsClassifier(n_neighbors=neighbors)  # 实例化KNN模型
        start_time = time.time()  # 记录开始时间
        clf.fit(train_features, train_label)  # 放入训练数据进行训练
        predict_label = clf.predict(test_features)  # 进行测试
        # print(predict_label == test_label)
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算程序运行时间，单位为秒
        # 将秒数转换为小时、分钟和秒数
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        # print(f"程序运行时间：{hours}小时 {minutes}分钟 {seconds}秒\n")
        # acc_sub = np.sum(predict_label == test_label) / len(test_label)
        test_score = clf.score(test_features, test_label)  # 预测分数
        test_acc += test_score
        # test_accArray.insert(k,test_score)
        test_accArray = np.append(test_accArray,test_score)
        # print(f'{k + 1}折 Test precision: ', test_score)
    # print("test_accArray:",test_accArray)
    # print("ave test acc:", test_acc / len(labels))
    return test_acc/len(labels), test_accArray


def EEGknn5(filename, model_path):
    path = "./data/SEED"
    path_list = os.listdir(path)
    total_dataTensor = torch.empty(0, 1, 62, 384)
    total_labelsTensor = torch.empty(0, 1)
    total_features = []
    total_labels = []
    print(len(path_list))
    norm_types = ['global_scale_value', 'global_gaussian_value', 'pixel_scale_value', 'origin']
    for i in range(0, len(path_list), 3):
        batch_files = path_list[i]
        # print(batch_files)
        norm_type = get_norm_type(filename, norm_types)
        X_data, X_label = get_dataset(batch_files, norm_type, 384, path)
        total_labels.append(X_label)
        X_data = X_data.astype('float32')
        X_data = torch.from_numpy(X_data)
        total_dataTensor = torch.cat((total_dataTensor, X_data), dim=0)
        # print(len(total_dataTensor), shape(total_dataTensor))
        labels = torch.Tensor(X_label)
        total_labelsTensor = torch.cat((total_labelsTensor, labels), dim=0)
        # print("第{}次".format((i / 3) + 1))
        i += 3
    # torch_dataset_get_feature=torch.from_numpy(total_dataTensor)
    torch_dataset_get_feature = Data.TensorDataset(total_dataTensor, total_labelsTensor)
    # total_features_2d, total_labels_1d  = get_feature(get_feature_dataloader)
    testaccmax = 0
    testacc_array_max = []
    total_results = []
    accmax = []
    for i in range(1):
        neighbors = 128
        # 64效果好
        get_feature_dataloader = Data.DataLoader(
            dataset=torch_dataset_get_feature,
            batch_size=128,
            shuffle=True,
            num_workers=0,
        )
        total_features_2d, total_labels_1d = extractor_cnn_rnn_para(get_feature_dataloader, 192, 1, filename, model_path)
        testacc, testacc_array = EEGknn(total_features_2d, total_labels_1d, total_labels, neighbors)
        if testacc - testaccmax > 0:
            testaccmax = testacc
            testacc_array_max = testacc_array
            m = i
        total_results.append(testacc_array_max)
        print("ave test acc:", testaccmax)
    return total_results

    #     get_feature_dataloader = Data.DataLoader(
    #         dataset=torch_dataset_get_feature,
    #         batch_size=128,
    #         shuffle=True,
    #         num_workers=0,
    #     )
    #     total_features_2d, total_labels_1d = extrator_cnn_rnn_para(get_feature_dataloader, 192, 1)
    #     testacc , testacc_array = EEGknn(total_features_2d, total_labels_1d, total_labels)
    # print("test_accArray:", testacc_array_max)
    # print("ave test acc:", testaccmax)


def get_filelist_knn(path):
    filenames = os.listdir(path)
    result = {}
    for filename in filenames:
        result_array = EEGknn5(filename, path)
        result[filename] = result_array
    return result


def get_file_knn(filename):
    result1 = EEGknn5(filename)

    print(filename + '的KNN准确率为:' + str(result1))

def knn_Seed():
    warnings.filterwarnings("ignore")
    model_path = './checkpoints/seed'
    result = get_filelist_knn(model_path)
    # get_file_knn('MSE+MI+Dis_sub0seedglobal_scale_value_mine_session1.pkl')
    print('end')





