import os
import time
import warnings
import mne
import random
import torch
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from torch.autograd import Variable
import torch.utils.data as Data
from models.Generator import EEGfuseNet_DEAP


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def extractor_cnn_rnn_para(loader_train, feature_size, label_dim, filename, model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EEGfuseNet_DEAP(16, 1, 1, 384).to(device)
    # model.load_state_dict(torch.load('generator_model.pth'))
    # model.load_state_dict(torch.load('MSE+MI_sub0seedglobal_scale_value_session1.pkl', map_location ='cpu'))
    # model.load_state_dict(torch.load('../result' + '/' + filename, map_location='cpu'))
    model.load_state_dict(torch.load(model_path + '/' + filename, map_location='cpu'))
    model.eval()
    code_mat = np.zeros((1, feature_size))
    label_mat = np.zeros((1, label_dim))
    with torch.no_grad():
        for step, (data, label) in enumerate(loader_train):
            inputs, label = Variable(data.to(device)), Variable(label.to(device))
            code, _, _ = model(inputs)
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


def EEGknn(total_feature, total_label, neighbors):
    kk = 32  # 留一交叉验证
    index1, index2 = 0, 2400
    test_acc = 0
    test_accArray = []
    test_accArray = np.array(test_accArray)
    for k in range(kk):
        step = 2400
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
        test_accArray = np.append(test_accArray, test_score)
        # print(f'{k + 1}折 Test precision: ', test_score)
    # print("test_accArray:",test_accArray)
    # print("ave test acc:", test_acc / len(labels))
    return test_acc/32, test_accArray


def EEGknn5(filename, model_path, emotion_dict,loader_datasets):

    testaccmax = 0
    testacc_array_max = []
    total_results = []
    accmax = []
    for i in range(1):
        neighbors = 64
        # 64效果好
        for j in range(len(loader_datasets)):
            total_features, total_labels = extractor_cnn_rnn_para(loader_datasets[j], 192, 1, filename, model_path)
            testacc, testacc_array = EEGknn(total_features, total_labels, neighbors)
            if testacc - testaccmax > 0:
                testaccmax = testacc
                testacc_array_max = testacc_array
                m = i
            total_results.append(testaccmax)
            print(f'{list(emotion_dict.keys())[j]} : ave test acc" {testaccmax}')
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


def get_TensorDataset_deap(emotion_dict):
    loader_datasets = []
    BATCH_SIZE = 128
    # 文件夹路径，包含MAT文件
    folder_path = 'F:\\repo\\Dual_MI\\Dual_MI\\data\\2_preprocessed_byJANE'
    # 初始化一个空的列表来存储所有矩阵
    all_matrices = []
    # 遍历文件夹中的所有MAT文件
    label_file = [f for f in os.listdir(folder_path) if 'deap_feature' in f][0]
    label_file = os.path.join(folder_path, label_file)
    label = np.array(loadmat(label_file)['label'])
    label[:, :4] = (label[:, :4] >= 5).astype(int)
    label = torch.from_numpy(label)
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.startswith('s') and file_name.endswith('.mat'):
            file_path = os.path.join(folder_path, file_name)
            mat_data = loadmat(file_path)
            if isinstance(mat_data['clean_data'], np.ndarray) and mat_data['clean_data'].dtype.names:
                for row in mat_data['clean_data']:
                    geneva_video = row['Geneva_video']
                    all_matrices.append(geneva_video)
            else:
                print(f"Warning: Unexpected data structure in {file_name}. Skipping.")
    all_matrices_array = np.array(all_matrices).reshape(1280)
    all_matrices_array = np.concatenate(all_matrices_array, axis=1)
    ch_names = ['eeg_' + str(i + 1) for i in range(32)]
    info = mne.create_info(
        # 通道名
        ch_names=ch_names,
        # 通道类型
        ch_types=['eeg' for _ in range(32)],
        # 采样频率
        sfreq=512
    )
    dataTensor = torch.empty(0, 1, 32, 384)
    data = all_matrices_array[:, :((np.shape(all_matrices_array)[1]) // 512) * 512]
    for i in range(5):
        sec_data = data[:, 256 * i * 30720:256 * (i + 1) * 30720]
        raw = mne.io.RawArray(sec_data, info)
        # 重采样
        raw = raw.copy().resample(sfreq=384)
        sec_data, _ = raw[:]
        sec_data = preprocessing.scale(sec_data)
        sec_data = sec_data.reshape(32, 1, np.shape(sec_data)[1] // 384, 384)
        sec_data = np.swapaxes(sec_data, 0, 2)
        sec_data = torch.tensor(sec_data, dtype=torch.float32)
        dataTensor = torch.cat([dataTensor, sec_data], dim=0)
    for i in range(len(emotion_dict)):
        torch_dataset = Data.TensorDataset(dataTensor, label[:, i:i+1])
        loader_dataset = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )
        loader_datasets.append(loader_dataset)
    return loader_datasets


def get_filelist_knn(path, emotion_dict):
    # filenames = os.listdir(path)
    filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    result = {}
    loader_datasets = get_TensorDataset_deap(emotion_dict)
    i = 1
    for filename in filenames:

        print(f'--------------sub{i}---------------')
        result_array = EEGknn5(filename, path, emotion_dict,loader_datasets)
        result[filename] = result_array
        i += 1
    return result


def get_file_knn(filename):
    result1 = EEGknn5(filename)
    print(filename + '的KNN准确率为:' + str(result1))


def knn_Deap():
    warnings.filterwarnings("ignore")
    emotion_dict = {
        "Valence": 0,
        "Arousal": 1,
        "Dominance": 2,
        "Liking": 3
    }
    model_path = 'F:\\repo\\Dual_MI\\Dual_MI\\checkpoints\\featurelist_train.py\\changedis'
    result = get_filelist_knn(model_path, emotion_dict)
    # get_file_knn('MSE+MI+Dis_sub0seedglobal_scale_value_mine_session1.pkl')
    print('end')





