import mne
import numpy as np
import torch
import torch.utils.data as Data
from scipy.io import loadmat
import os
from sklearn import preprocessing


def get_TensorDataset_deap(sub):
    BATCH_SIZE = 128
    # 文件夹路径，包含MAT文件
    folder_path = '../data/2_preprocessed_byJANE'
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
    train_dataTensor = dataTensor[len(dataTensor) // 32:]
    valid_dataTensor = dataTensor[0:len(dataTensor) // 32]
    train_labelsTensor = label[len(label) // 32:]
    valid_labelsTensor = label[0:len(label) // 32]
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