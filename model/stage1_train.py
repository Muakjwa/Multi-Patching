import AutoEncoder
import numpy as np
import torch

compress_minute = 40
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
val_data = np.load('../data_npy/val/val_hr.npy')[:, 0, :][:, np.newaxis, :]
test_data = np.load('../data_npy/test/test_hr.npy')[:, 0, :][:, np.newaxis, :]
compressed_val_data, compressed_test_data = AutoEncoder.AutoEncoding_nfill_VT(4, device, val_data, test_data, f'{compress_minute}_hr', 1/60, compress_minute, False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for i in range(7):
    val_data = np.load('../data_npy/val/val_pedo.npy')[:, i, :][:, np.newaxis, :]
    test_data = np.load('../data_npy/test/test_pedo.npy')[:, i, :][:, np.newaxis, :]
    compressed_val_data, compressed_test_data = AutoEncoder.AutoEncoding_nfill_VT(4, device, val_data, test_data, f'{compress_minute}_pedo_{i}', 1/60, compress_minute, False)