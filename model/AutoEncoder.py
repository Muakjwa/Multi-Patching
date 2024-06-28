import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy.interpolate import CubicSpline


# 오토인코더 정의
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, compressed_dim):
        super(LSTMEncoder, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, compressed_dim)
        
    def forward(self, x):
        _, (h_n, _) = self.lstm1(x)
        h_n = h_n[-1]
        out = self.fc(h_n)
        return out

class LSTMDecoder(nn.Module):
    def __init__(self, compressed_dim, hidden_dim, output_dim, seq_len):
        super(LSTMDecoder, self).__init__()
        self.fc = nn.Linear(compressed_dim, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.seq_len = seq_len
        
    def forward(self, x):
        x = self.fc(x).unsqueeze(1).repeat(1, self.seq_len, 1)
        x, _ = self.lstm1(x)
        out = self.fc_out(x)
        return out

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, compressed_dim, seq_len):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, compressed_dim)
        self.decoder = LSTMDecoder(compressed_dim, hidden_dim, input_dim, seq_len)
        
    def forward(self, x):
        compressed = self.encoder(x)
        reconstructed = self.decoder(compressed)
        return reconstructed

# 데이터셋 및 데이터로더 정의
class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# NaN 값을 평균값으로 채우는 함수
def fill_nans_with_mean(segment):
    segment = segment.squeeze(0)
    for feature in range(segment.shape[1]):
        nan_indices = np.isnan(segment[:, feature])
        if np.any(nan_indices):
            non_nan_indices = ~nan_indices
            if np.sum(non_nan_indices) == 0:  # 모든 값이 NaN인 경우
                return None
            mean_value = np.nanmean(segment[:, feature])
            segment[nan_indices, feature] = mean_value
    return segment

# NaN 값을 Spline 보간으로 채우는 함수
def fill_nans_with_spline(segment):
    segment = segment.squeeze(0)
    for feature in range(segment.shape[1]):
        nan_indices = np.isnan(segment[:, feature])
        if np.any(nan_indices):
            non_nan_indices = ~nan_indices
            if np.sum(non_nan_indices) < 2:  # 유효한 데이터가 두 개 미만인 경우
                return None
            
            # Get the indices of non-NaN values and the corresponding values
            x_non_nan = np.arange(segment.shape[0])[non_nan_indices]
            y_non_nan = segment[non_nan_indices, feature]
            
            # Create the spline interpolation
            spline_interp = CubicSpline(x_non_nan, y_non_nan)
            
            # Interpolate the NaN values
            x_nan = np.arange(segment.shape[0])[nan_indices]
            segment[nan_indices, feature] = spline_interp(x_nan)
    return segment

# 데이터를 20분 단위로 분할
def split_data(data, segment_samples):
    num_samples, num_features, total_length = data.shape
    num_segments = total_length // segment_samples
    segments = []

    for sample in range(num_samples):
        for segment in range(num_segments):
            seg_data = data[sample, :, segment * segment_samples:(segment + 1) * segment_samples]
            segments.append(seg_data)

    return np.array(segments)


def add_white_noise(data, noise_level=0.05):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise


def process_and_save_data(model, data, segment_samples, num_features, train_data_path, sample_minute, device, file_prefix, isSpline = False):
    # 데이터 처리 및 분할
    segments = split_data(data, segment_samples)
    segments = segments.reshape(-1, segment_samples, num_features)
    eval_segments = torch.tensor(segments, dtype=torch.float32).to(device)
    
    model.eval()
    compressed_full_data = []
    print(eval_segments.shape)
    with torch.no_grad():
        for sample in eval_segments:
            sample = sample.unsqueeze(0).to(device)  # (1, segment_samples, num_features)
            if (~np.isnan(sample.cpu().numpy()).sum()) > 0: # <= (segment_samples * num_features * nan_threshold):
                if (isSpline):
                    sample = fill_nans_with_spline(sample.cpu().numpy())
                else:
                    sample = fill_nans_with_mean(sample.cpu().numpy())

                if sample is None:
                    compressed_full_data.append(np.full((1, (int(16/sample_minute))), np.nan))
                    continue
                sample = torch.tensor(sample, dtype=torch.float32).to(device)
            compressed_segment = model.encoder(sample).cpu().numpy().reshape(1, -1)
            compressed_full_data.append(compressed_segment)
    
    compressed_full_data = np.array(compressed_full_data)  # (num_samples, compressed_length)
    compressed_full_data = compressed_full_data.transpose(0, 2, 1)  # (num_samples, compressed_length, num_segments)
    compressed_full_data = compressed_full_data.reshape(data.shape[0], num_features, -1)  # (471, 1, 1440)

    torch.save(model, f'../data_npy/model/AE_{file_prefix}_{int(1440/sample_minute)}_{train_data_path}_model.pt')

    print(f"{file_prefix.capitalize()} data compressed shape:", compressed_full_data.shape)
    return compressed_full_data

    
def AutoEncoding_nfill_VT(sample_minute, device, val_data, test_data, train_data_path, sampling_rate, compress_minute, isSpline = False, noise = 0, noise_level = 0.1):
    print(f"Using device: {device}")

    segment_length_seconds = compress_minute * 60  # compresse minute * 60 분
    segment_samples = int(segment_length_seconds * sampling_rate)  # Sample 수
    compressed_length = int(compress_minute / sample_minute)  # 압축된 데이터 길이
    num_features = val_data.shape[1]

    # 모델 생성
    input_dim = num_features
    hidden_dim = compress_minute * 2
    compressed_dim = compressed_length
    seq_len = segment_samples

    model = LSTMAutoencoder(input_dim, hidden_dim, compressed_dim, seq_len).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

    # 학습 데이터 준비
    train_segments = split_data(np.concatenate((val_data, test_data), axis=0), segment_samples)
    train_segments = train_segments.reshape(-1, segment_samples, num_features)
    if (noise):
        augmented_data = []
        for _ in range(noise):
            new_data_noised = add_white_noise(train_segments, noise_level)
            augmented_data.append(new_data_noised)
        train_data = np.concatenate(augmented_data, axis=0)
        train_segments = torch.tensor(train_data, dtype=torch.float32).to(device)
    else:
        train_segments = torch.tensor(train_segments, dtype=torch.float32).to(device)
    train_new_segments = []
    
    for i, sample in enumerate(train_segments):
        sample = sample.unsqueeze(0).to(device) 
        if (~np.isnan(sample.cpu().numpy())).sum() > 0:
            if (isSpline):
                sample = fill_nans_with_spline(sample.cpu().numpy())
            else:
                sample = fill_nans_with_mean(sample.cpu().numpy())

            if sample is None:
                continue
            train_new_segments.append(torch.tensor(sample, dtype=torch.float32).to(device))

    train_dataset = TimeSeriesDataset(train_new_segments)
    train_dataloader = DataLoader(train_dataset, batch_size=16384, shuffle=True)

    # 모델 학습
    num_epochs = 10000
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        
        if ((epoch+1)%50 == 0):
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_dataloader):.4f}")

    # 데이터 압축 및 저장
    compressed_val_data = process_and_save_data(model, val_data, segment_samples, num_features, train_data_path, sample_minute, device, 'val', isSpline)
    compressed_test_data = process_and_save_data(model, test_data, segment_samples, num_features, train_data_path, sample_minute, device, 'test', isSpline)

    return compressed_val_data, compressed_test_data


def AutoEncoding_nfill_VT_Test(model, sample_minute, device, val_data, test_data, train_data_path, sampling_rate, compress_minute, isSpline = False):
    segment_length_seconds = compress_minute * 60  # compresse minute * 60 분
    segment_samples = int(segment_length_seconds * sampling_rate)  # Sample 수

    num_features = test_data.shape[1]
    
    compressed_val_data = process_and_save_data(model, val_data, segment_samples, num_features, train_data_path, sample_minute, device, 'val', isSpline)
    
    # 압축된 데이터 저장
    compressed_data_path = f'../data_npy/encoded/AE_val_compressed_{int(1440/sample_minute)}_{train_data_path}'
    np.save(compressed_data_path, compressed_val_data)
    print(f"VAL data saved to: {compressed_data_path}")

    compressed_test_data = process_and_save_data(model, test_data, segment_samples, num_features, train_data_path, sample_minute, device, 'test', isSpline)
    
    # 압축된 데이터 저장
    compressed_data_path = f'../data_npy/encoded/AE_test_compressed_{int(1440/sample_minute)}_{train_data_path}'
    np.save(compressed_data_path, compressed_test_data)
    print(f"TEST data saved to: {compressed_data_path}")

    return compressed_val_data, compressed_test_data