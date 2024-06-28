import numpy as np
from scipy.interpolate import CubicSpline
import numpy as np
import pandas as pd
import torch
import PatchTST

def expand_time_series(data):
    original_time = np.linspace(0, 1, data.shape[2])
    new_time = np.linspace(0, 1, 360)
    
    new_data = np.zeros((data.shape[0], data.shape[1], 360))
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            spline_interp = CubicSpline(original_time, data[i, j])
            new_data[i, j] = spline_interp(new_time)
    return new_data

file_path = '../data_npy/encoded/'
model_path = '../data_npy/model/patchTST_6noise(5)_360_16'

labels = ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4']
train_data1 = np.load(file_path + 'AE_val_compressed_360_40_hr.npy', allow_pickle=True)
train_data2 = np.load(file_path + 'AE_val_compressed_360_40_pedo_0.npy', allow_pickle=True)
train_data3 = np.load(file_path + 'AE_val_compressed_360_40_pedo_1.npy', allow_pickle=True)
train_data4 = np.load(file_path + 'AE_val_compressed_360_40_pedo_2.npy', allow_pickle=True)
train_data5 = np.load(file_path + 'AE_val_compressed_360_40_pedo_3.npy', allow_pickle=True)
train_data6 = np.load(file_path + 'AE_val_compressed_360_40_pedo_4.npy', allow_pickle=True)
train_data7 = np.load(file_path + 'AE_val_compressed_360_40_pedo_5.npy', allow_pickle=True)
train_data8 = np.load(file_path + 'AE_val_compressed_360_40_pedo_6.npy', allow_pickle=True)
train_data9 = expand_time_series(np.load('../data_npy/val/val_m_light.npy', allow_pickle=True))
train_data10 = expand_time_series(np.load('../data_npy/val/val_w_light.npy', allow_pickle=True))
train_data = np.concatenate((train_data1, train_data2, train_data3, train_data4, train_data5, train_data6, train_data7, train_data8, train_data9, train_data10), axis = 1)

test_data1 = np.load(file_path + 'AE_test_compressed_360_40_hr.npy', allow_pickle=True)
test_data2 = np.load(file_path + 'AE_test_compressed_360_40_pedo_0.npy', allow_pickle=True)
test_data3 = np.load(file_path + 'AE_test_compressed_360_40_pedo_1.npy', allow_pickle=True)
test_data4 = np.load(file_path + 'AE_test_compressed_360_40_pedo_2.npy', allow_pickle=True)
test_data5 = np.load(file_path + 'AE_test_compressed_360_40_pedo_3.npy', allow_pickle=True)
test_data6 = np.load(file_path + 'AE_test_compressed_360_40_pedo_4.npy', allow_pickle=True)
test_data7 = np.load(file_path + 'AE_test_compressed_360_40_pedo_5.npy', allow_pickle=True)
test_data8 = np.load(file_path + 'AE_test_compressed_360_40_pedo_6.npy', allow_pickle=True)
test_data9 = expand_time_series(np.load('../data_npy/test/test_m_light.npy', allow_pickle=True))
test_data10 = expand_time_series(np.load('../data_npy/test/test_w_light.npy', allow_pickle=True))
test_data = np.concatenate((test_data1, test_data2, test_data3, test_data4, test_data5, test_data6, test_data7, test_data8, test_data9, test_data10), axis = 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
answer_path = '../answer.csv'

for i, label in enumerate(labels):
    model = torch.load(model_path + label + '.pt').to(device)
    y_pred = PatchTST.patchTST_AE_test(model, test_data, device, 8)
    answer_activity = pd.read_csv(answer_path)
    answer_activity[label] = (np.array(y_pred).astype(np.int32))
    answer_activity.to_csv(answer_path, index=False)