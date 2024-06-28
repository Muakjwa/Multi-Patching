import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from transformers import PatchTSTConfig, PatchTSTForClassification


def add_white_noise(data, noise_level=0.05):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise


# NaN 처리 및 마스크 생성
def create_mask_and_replace_nan(data):
    mask = ~pd.isna(data)
    data[pd.isna(data)] = 0  # NaN 값을 0으로 대체
    return data, mask

def patchTST_CV_AE(train_data, train_labels, device, patch_length, epoches, batch_size):
    # # 데이터 로드
    # train_data = np.load(data_path, allow_pickle=True)
    # train_labels = np.load(label_path)

    # 데이터 차원 변환: (samples, features, time_series_length) -> (samples, time_series_length, features)
    train_data = np.transpose(train_data, (0, 2, 1))

    # train_data = train_data[:, :, 5][:, :, np.newaxis]
    # 데이터 형태 확인
    print(f"Train data shape: {train_data.shape}")
    print(f"Train labels shape: {train_labels.shape}")

    # NaN 처리 및 마스크 생성
    train_data_np = train_data
    train_mask = ~np.isnan(train_data_np)
    train_data_np[np.isnan(train_data_np)] = 0  # NaN 값을 0으로 대체

    # PyTorch 텐서로 변환
    train_data = torch.tensor(train_data_np, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    train_mask = torch.tensor(train_mask, dtype=torch.bool)

    print(train_data.shape)

    # 시퀀스 길이 및 특징 수 확인
    sequence_length = train_data.shape[1]
    num_input_channels = train_data.shape[2]

    print(f"Configured sequence length: {sequence_length}")
    print(f"Number of input channels: {num_input_channels}")

    # PatchTST 모델 설정
    config = PatchTSTConfig(
        num_input_channels=num_input_channels,  # number of features
        context_length=sequence_length,  # explicitly set sequence length based on input data
        num_targets=2,  # binary classification
        patch_length=patch_length,
        stride=patch_length/2,
        use_cls_token=True,
    )

    # 교차 검증 설정
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []

    for fold, (train_index, val_index) in enumerate(kf.split(train_data, train_labels)):
        print(f'Fold {fold + 1}')
        
        # 데이터셋 및 데이터로더 설정
        train_subset = Subset(TensorDataset(train_data, train_labels, train_mask), train_index)
        val_subset = Subset(TensorDataset(train_data, train_labels, train_mask), val_index)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        # 모델 초기화
        model = PatchTSTForClassification(config).to(device)
        
        # 손실 함수와 옵티마이저 정의
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        
        # 학습 루프
        epochs = epoches
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch_data, batch_labels, batch_mask in train_loader:
                batch_data, batch_labels, batch_mask = batch_data.to(device), batch_labels.to(device), batch_mask.to(device)

                optimizer.zero_grad()
                outputs = model(past_values=batch_data, past_observed_mask=batch_mask)
                loss = criterion(outputs.prediction_logits, batch_labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            if ((epoch+1) % 50 == 0):
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss}')

            # # Early stopping 조건
            # if avg_epoch_loss < 0.01:
            #     print(f'Early stopping at epoch {epoch+1} with loss {avg_epoch_loss}')
            #     break
        
        # 검증 데이터에 대한 예측 및 F1-score 계산
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_data, batch_labels, batch_mask in val_loader:
                batch_data, batch_labels, batch_mask = batch_data.to(device), batch_labels.to(device), batch_mask.to(device)
                outputs = model(past_values=batch_data, past_observed_mask=batch_mask)
                _, predicted = torch.max(outputs.prediction_logits, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        f1 = f1_score(all_labels, all_preds, average='macro')
        f1_scores.append(f1)
        print(f'Fold {fold + 1} F1-score: {f1}')

    # 평균 F1-score 출력
    mean_f1 = np.mean(f1_scores)
    print(f'Mean F1-score: {mean_f1}')

def patchTST_CV_sampling(feature, train_data, train_labels, device, patch_length, epoches, batch_size):
    # # 데이터 로드
    # train_data = np.load(data_path, allow_pickle=True)
    # train_labels = np.load(label_path)

    feature_dict = {'x' : 0, 'y' : 1, 'z' : 2, 'lat' : 3, 'lon' : 4, 'hr' : 5}
    
    if feature in feature_dict:
        train_data = train_data[:, :, feature_dict[feature]][:, :, np.newaxis]
    # 데이터 형태 확인
    print(f"Train data shape: {train_data.shape}")
    print(f"Train labels shape: {train_labels.shape}")

    # NaN 처리: 평균값으로 대체 후, 남은 NaN 값은 0으로 대체
    def nan_mean_fill(data):
        """평균값으로 NaN 값을 대체하고 남은 NaN 값을 0으로 대체"""
        nan_mean = np.nanmean(data, axis=1, keepdims=True)
        data = np.where(np.isnan(data), nan_mean, data)
        data = np.nan_to_num(data)  # 남아 있는 NaN 값을 0으로 대체
        return data

    # NaN 값을 평균값으로 대체 후 남은 NaN 값을 0으로 대체
    train_data_np = nan_mean_fill(train_data)

    # PyTorch 텐서로 변환
    train_data = torch.tensor(train_data_np, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)

    # 시퀀스 길이 및 특징 수 확인
    sequence_length = train_data.shape[1]
    num_input_channels = train_data.shape[2]

    print(f"Configured sequence length: {sequence_length}")
    print(f"Number of input channels: {num_input_channels}")

    # PatchTST 모델 설정
    config = PatchTSTConfig(
        num_input_channels=num_input_channels,  # number of features
        context_length=sequence_length,  # explicitly set sequence length based on input data
        num_targets=2,  # binary classification
        patch_length=patch_length,
        stride=patch_length/2,
        use_cls_token=True,
    )

    # 교차 검증 설정
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []

    for fold, (train_index, val_index) in enumerate(kf.split(train_data, train_labels)):
        print(f'Fold {fold + 1}')
        
        # 데이터셋 및 데이터로더 설정
        train_subset = Subset(TensorDataset(train_data, train_labels), train_index)
        val_subset = Subset(TensorDataset(train_data, train_labels), val_index)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        # 모델 초기화
        model = PatchTSTForClassification(config).to(device)
        
        # 손실 함수와 옵티마이저 정의
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        
        # 학습 루프
        epochs = epoches
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch_data, batch_labels in train_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(past_values=batch_data)
                loss = criterion(outputs.prediction_logits, batch_labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
            
            if ((epoch+1) % 50 == 0):
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader)}')
        
        # 검증 데이터에 대한 예측 및 F1-score 계산
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                outputs = model(past_values=batch_data)
                _, predicted = torch.max(outputs.prediction_logits, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        f1 = f1_score(all_labels, all_preds, average='macro')
        f1_scores.append(f1)
        print(f'Fold {fold + 1} F1-score: {f1}')

    # 평균 F1-score 출력
    mean_f1 = np.mean(f1_scores)
    print(f'Mean F1-score: {mean_f1}')


def patchTST_AE(train_data, train_labels, val_data, val_labels, device, patch_length, epochs, batch_size):
    # # 데이터 로드
    # train_data = np.load(train_data_path, allow_pickle=True)
    # train_labels = np.load(train_label_path)
    # val_data = np.load(val_data_path, allow_pickle=True)
    # val_labels = np.load(val_label_path)

    # 데이터 차원 변환: (samples, features, time_series_length) -> (samples, time_series_length, features)
    train_data = np.transpose(train_data, (0, 2, 1))
    val_data = np.transpose(val_data, (0, 2, 1))

    train_data_np, train_mask = create_mask_and_replace_nan(train_data)
    val_data_np, val_mask = create_mask_and_replace_nan(val_data)

    # PyTorch 텐서로 변환
    train_data = torch.tensor(train_data_np, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    train_mask = torch.tensor(train_mask, dtype=torch.bool)

    val_data = torch.tensor(val_data_np, dtype=torch.float32)
    val_labels = torch.tensor(val_labels, dtype=torch.long)
    val_mask = torch.tensor(val_mask, dtype=torch.bool)

    # 시퀀스 길이 및 특징 수 확인
    sequence_length = train_data.shape[1]
    num_input_channels = train_data.shape[2]

    print(f"Configured sequence length: {sequence_length}")
    print(f"Number of input channels: {num_input_channels}")

    # PatchTST 모델 설정
    config = PatchTSTConfig(
        num_input_channels=num_input_channels,  # number of features
        context_length=sequence_length,  # explicitly set sequence length based on input data
        num_targets=2,  # binary classification
        patch_length=patch_length,
        stride=patch_length // 2,
        use_cls_token=True,
    )

    # 데이터셋 및 데이터로더 설정
    train_dataset = TensorDataset(train_data, train_labels, train_mask)
    val_dataset = TensorDataset(val_data, val_labels, val_mask)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 모델 초기화
    model = PatchTSTForClassification(config).to(device)
    
    # 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    
    # 학습 루프
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_data, batch_labels, batch_mask in train_loader:
            batch_data, batch_labels, batch_mask = batch_data.to(device), batch_labels.to(device), batch_mask.to(device)

            optimizer.zero_grad()
            outputs = model(past_values=batch_data, past_observed_mask=batch_mask)
            loss = criterion(outputs.prediction_logits, batch_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        
        if ((epoch+1) % 50 == 0):
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss}')

        # Early stopping 조건
        if avg_epoch_loss < 0.01:
            print(f'Early stopping at epoch {epoch+1} with loss {avg_epoch_loss}')
            break
    
    # 검증 데이터에 대한 예측 및 F1-score 계산
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_data, batch_labels, batch_mask in val_loader:
            batch_data, batch_labels, batch_mask = batch_data.to(device), batch_labels.to(device), batch_mask.to(device)
            outputs = model(past_values=batch_data, past_observed_mask=batch_mask)
            _, predicted = torch.max(outputs.prediction_logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f'Validation F1-score: {f1}')

def patchTST_AE_train(train_data, train_labels, device, patch_length, epochs, batch_size, features):
    # 데이터 차원 변환: (samples, features, time_series_length) -> (samples, time_series_length, features)
    train_data = np.transpose(train_data, (0, 2, 1))

    train_data_np, train_mask = create_mask_and_replace_nan(train_data)

    # PyTorch 텐서로 변환
    train_data = torch.tensor(train_data_np, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    train_mask = torch.tensor(train_mask, dtype=torch.bool)

    # 시퀀스 길이 및 특징 수 확인
    sequence_length = train_data.shape[1]
    num_input_channels = train_data.shape[2]

    print(f"Configured sequence length: {sequence_length}")
    print(f"Number of input channels: {num_input_channels}")

    # PatchTST 모델 설정
    config = PatchTSTConfig(
        num_input_channels=num_input_channels,  # number of features
        context_length=sequence_length,  # explicitly set sequence length based on input data
        num_targets=2,  # binary classification
        patch_length=patch_length,
        stride=patch_length // 2,
        use_cls_token=True,
    )

    # 데이터셋 및 데이터로더 설정
    train_dataset = TensorDataset(train_data, train_labels, train_mask)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 모델 초기화
    model = PatchTSTForClassification(config).to(device)
    
    # 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    
    # 학습 루프
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_data, batch_labels, batch_mask in train_loader:
            batch_data, batch_labels, batch_mask = batch_data.to(device), batch_labels.to(device), batch_mask.to(device)

            optimizer.zero_grad()
            outputs = model(past_values=batch_data, past_observed_mask=batch_mask)
            loss = criterion(outputs.prediction_logits, batch_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)        
        if ((epoch+1) % 50 == 0):
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss}')

        # Early stopping 조건
        # if avg_epoch_loss < 0.01:
        #     print(f'Early stopping at epoch {epoch+1} with loss {avg_epoch_loss}')
        #     break
    
    path = f'./data_npy/model/patchTST_{features}.pt'
    torch.save(model, path)
    print(f'[PatchTST Train Complete!] : {path}')
    return model

def patchTST_AE_test(model, test_data, device, batch_size):
    # 데이터 차원 변환: (samples, features, time_series_length) -> (samples, time_series_length, features)
    test_data = np.transpose(test_data, (0, 2, 1))

    test_data_np, test_mask = create_mask_and_replace_nan(test_data)

    # PyTorch 텐서로 변환
    test_data = torch.tensor(test_data_np, dtype=torch.float32)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)

    # 데이터셋 및 데이터로더 설정
    test_dataset = TensorDataset(test_data, test_mask)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 검증 데이터에 대한 예측 및 F1-score 계산
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch_data, batch_mask in test_loader:
            batch_data, batch_mask = batch_data.to(device), batch_mask.to(device)
            outputs = model(past_values=batch_data, past_observed_mask=batch_mask)
            _, predicted = torch.max(outputs.prediction_logits, 1)
            all_preds.extend(predicted.cpu().numpy())
    
    print(f'[Inference Complete!]')
    return all_preds

def patchTST_pedo_train(train_data, train_labels, device, patch_length, epochs, batch_size, features):
    # 데이터 형태 확인
    print(f"Train data shape: {train_data.shape}")
    print(f"Train labels shape: {train_labels.shape}")

    # 데이터 차원 변환: (samples, features, time_series_length) -> (samples, time_series_length, features)
    train_data = np.transpose(train_data, (0, 2, 1))

    # PyTorch 텐서로 변환
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)

    # 시퀀스 길이 및 특징 수 확인
    sequence_length = train_data.shape[1]
    num_input_channels = train_data.shape[2]

    print(f"Configured sequence length: {sequence_length}")
    print(f"Number of input channels: {num_input_channels}")

    # PatchTST 모델 설정
    config = PatchTSTConfig(
        num_input_channels=num_input_channels,  # number of features
        context_length=sequence_length,  # explicitly set sequence length based on input data
        num_targets=2,  # binary classification
        patch_length=patch_length,
        stride=patch_length/2,
        use_cls_token=True,
    )

    model = PatchTSTForClassification(config).to(device)  # 각 epoch마다 모델 초기화 및 GPU로 이동

    # 데이터셋 및 데이터로더 설정
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

    # 학습 루프
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs.prediction_logits, batch_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
               
        if ((epoch+1) % 50 == 0):
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader)}')

    path = f'./data_npy/model/patchTST_{features}.pt'
    torch.save(model, path)
    print(f'[PatchTST Train Complete!] : {path}')
    return model

def patchTST_pedo_test(model, test_data, device, batch_size):
    print(f"Test data shape: {test_data.shape}")

    test_data = np.transpose(test_data, (0, 2, 1))
    test_data = torch.tensor(test_data, dtype=torch.float32)

    test_dataset = TensorDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 테스트 데이터에 대한 예측
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data[0].to(device)  # batch_data는 (data,) 형태로 튜플로 반환됨
            outputs = model(batch_data)
            _, predicted = torch.max(outputs.prediction_logits, 1)
            all_preds.extend(predicted.cpu().numpy())

    # 예측 결과 출력
    print("Predicted labels for the test data:", all_preds)
    return all_preds


def patchTST_AE_train_noise(train_data, train_labels, device, patch_length, epochs, batch_size, features, noise_level=0.05, augment_factor=1):
    # 데이터 차원 변환: (samples, features, time_series_length) -> (samples, time_series_length, features)
    train_data = np.transpose(train_data, (0, 2, 1))
    
    # 데이터 증강
    augmented_data = []
    augmented_labels = []
    for _ in range(augment_factor):
        new_data_noised = add_white_noise(train_data, noise_level)
        augmented_data.append(new_data_noised)
        augmented_labels.append(train_labels)
    
    train_data = np.concatenate(augmented_data, axis=0)
    train_labels = np.concatenate(augmented_labels, axis=0)
    
    train_data_np, train_mask = create_mask_and_replace_nan(train_data)

    # PyTorch 텐서로 변환
    train_data = torch.tensor(train_data_np, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    train_mask = torch.tensor(train_mask, dtype=torch.bool)

    # 시퀀스 길이 및 특징 수 확인
    sequence_length = train_data.shape[1]
    num_input_channels = train_data.shape[2]

    print(f"Configured sequence length: {sequence_length}")
    print(f"Number of input channels: {num_input_channels}")

    # PatchTST 모델 설정
    config = PatchTSTConfig(
        num_input_channels=num_input_channels,  # number of features
        context_length=sequence_length,  # explicitly set sequence length based on input data
        num_targets=2,  # binary classification
        patch_length=patch_length,
        stride=patch_length // 2,
        use_cls_token=True,
    )

    # 데이터셋 및 데이터로더 설정
    train_dataset = TensorDataset(train_data, train_labels, train_mask)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 모델 초기화
    model = PatchTSTForClassification(config).to(device)
    
    # 손실 함수와 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    
    # 학습 루프
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_data, batch_labels, batch_mask in train_loader:
            batch_data, batch_labels, batch_mask = batch_data.to(device), batch_labels.to(device), batch_mask.to(device)

            optimizer.zero_grad()
            outputs = model(past_values=batch_data, past_observed_mask=batch_mask)
            loss = criterion(outputs.prediction_logits, batch_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)        
        if ((epoch+1) % 50 == 0):
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss}')

        # Early stopping 조건
        # if avg_epoch_loss < 0.01:
        #     print(f'Early stopping at epoch {epoch+1} with loss {avg_epoch_loss}')
        #     break
    
    path = f'../data_npy/model/patchTST_{features}.pt'
    torch.save(model, path)
    print(f'[PatchTST Train Complete!] : {path}')
    return model

