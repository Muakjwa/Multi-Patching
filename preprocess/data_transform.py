import numpy as np
import pandas as pd
import datetime
from datetime import datetime
import time
import torch


""" DATA PREPROCESSING FOR VALIDATION AND TEST DATASET """

# isvalid : Preprocessing Data가 val인지 test인지에 대한 parameter
# pos     : Preprocessed Data 저장 위치 parameter
# feat    : gps와 hr는 같은 함수를 사용 (어떤 feature를 추출할지에 대한 parameter)

def ext_acc(isval, pos):
    start = time.time()
    unit = 100

    # DATAFRAME LOAD
    if (isval):
        df_acc_1 = pd.read_parquet('../data/val_dataset/ch2024_val__m_acc_part_1.parquet.gzip')
        df_acc_2 = pd.read_parquet('../data/val_dataset/ch2024_val__m_acc_part_2.parquet.gzip')
        df_acc_3 = pd.read_parquet('../data/val_dataset/ch2024_val__m_acc_part_3.parquet.gzip')
        df_acc_4 = pd.read_parquet('../data/val_dataset/ch2024_val__m_acc_part_4.parquet.gzip')
        val_label = pd.read_csv('../data/val_label.csv')
        file_name = 'val_acc.npy'
    else:
        df_acc_1 = pd.read_parquet('../data/test_dataset/ch2024_test__m_acc_part_5.parquet.gzip')
        df_acc_2 = pd.read_parquet('../data/test_dataset/ch2024_test__m_acc_part_6.parquet.gzip')
        df_acc_3 = pd.read_parquet('../data/test_dataset/ch2024_test__m_acc_part_7.parquet.gzip')
        df_acc_4 = pd.read_parquet('../data/test_dataset/ch2024_test__m_acc_part_8.parquet.gzip')
        val_label = pd.read_csv('../data/answer_sample.csv')
        file_name = 'test_acc.npy'
    
    data_list = []
    date_format = "%Y-%m-%d"
    print(f"[DataFrame Generation Finish] : {time.time() - start:.5f} sec")

    # UNIX TIME FOR INDEX; (SOME COLUMN NAME EDITING)
    val_label['date'] = val_label['date'].apply(lambda x : (int(datetime.strptime(x, date_format).timestamp()+(9*60*60)) * 1000)//(unit)*(unit) if True else x)
    df_list = [df_acc_1, df_acc_2, df_acc_3, df_acc_4]
    for df in df_list:
        df['timestamp'] = df['timestamp'].apply(lambda x : (int(x.timestamp()+(9*60*60)) * 1000)//(unit)*(unit) if True else x)
        df.drop_duplicates(subset = ['timestamp'], inplace = True)
        df.set_index('timestamp', inplace = True)

    features = ['x', 'y', 'z']
    df_acc = [df_acc_1, df_acc_2, df_acc_3, df_acc_4]
    df_nacc = []
    print(f"[DataFrame Editing Finish] : {time.time() - start:.5f} sec")
            
    for i, date in enumerate(val_label['date']):
        temp_list = []
        user = val_label.loc[i, 'subject_id']
        index = range(date, date + (24*60*60*1000), unit)
        data = [[float('nan')] * len(features)] * len(index)
        total_df = pd.DataFrame(data, index=index, columns=features)
        for df in df_nacc:
            total_df = total_df.combine_first(df[df['subject_id'] == user][list(set(df.columns) & set(features))]).loc[index, :]
        if (isval):
            total_df = total_df.combine_first(df_acc[user-1][list(set(df_acc[user-1].columns) & set(features))]).loc[index, :]
        else:
            total_df = total_df.combine_first(df_acc[user-5][list(set(df_acc[user-5].columns) & set(features))]).loc[index, :]
        total_df.ffill(inplace = True)
        
        for feature in features:
            temp_list.append(list(total_df[feature]))
        data_list.append(temp_list)

    print(f"Total Time : {time.time() - start:.5f} sec")

    final_tensor = np.array(data_list)

    np.save(pos + file_name, final_tensor)


def ext_gps_hr(isval, pos, feat = 'gps'):
    start = time.time()

    # DATAFRAME LOAD
    if (isval):
        df_hr = pd.read_parquet('../data/val_dataset/ch2024_val__w_heart_rate.parquet.gzip')
        df_gps = pd.read_parquet('../data/val_dataset/ch2024_val__m_gps.parquet.gzip')
        val_label = pd.read_csv('../data/val_label.csv')
    else:
        df_hr = pd.read_parquet('../data/test_dataset/ch2024_test_w_heart_rate.parquet.gzip')
        df_gps = pd.read_parquet('../data/test_dataset/ch2024_test_m_gps.parquet.gzip')
        val_label = pd.read_csv('../data/answer_sample.csv')
    data_list = []
    date_format = "%Y-%m-%d"
    print(f"[DataFrame Generation Finish] : {time.time() - start:.5f} sec")

    if feat == 'gps':
        df_list = [df_gps]
        unit = 5000
        if (isval):
            file_name = 'val_gps.npy'
        else:
            file_name = 'test_gps.npy'
    else:
        df_list = [df_hr]
        unit = 60000
        if (isval):
            file_name = 'val_hr.npy'
        else:
            file_name = 'test_hr.npy'

    # UNIX TIME FOR INDEX; (SOME COLUMN NAME EDITING)
    val_label['date'] = val_label['date'].apply(lambda x : (int(datetime.strptime(x, date_format).timestamp()+(9*60*60)) * 1000)//(unit)*(unit) if True else x)

    for df in df_list:
        df['timestamp'] = df['timestamp'].apply(lambda x : (int(x.timestamp()+(9*60*60)) * 1000)//(unit)*(unit) if True else x)
        df.drop_duplicates(subset = ['timestamp'], inplace = True)
        df.set_index('timestamp', inplace = True)
        if ('latitude' in df.columns):
            df.rename(columns = {'latitude' : 'lat', 'longitude' : 'lon'}, inplace = True)
        elif ('heart_rate' in df.columns):
            df.rename(columns = {'heart_rate' : 'hr'}, inplace = True)

    if feat == 'gps':
        features = ['lat', 'lon']
    else:
        features = ['hr']

    df_nacc = [df_hr, df_gps]
    print(f"[DataFrame Editing Finish] : {time.time() - start:.5f} sec")
            
    for i, date in enumerate(val_label['date']):
        temp_list = []
        user = val_label.loc[i, 'subject_id']
        index = range(date, date + (24*60*60*1000), unit)
        data = [[float('nan')] * len(features)] * len(index)
        total_df = pd.DataFrame(data, index=index, columns=features)
        for df in df_nacc:
            total_df = total_df.combine_first(df[df['subject_id'] == user][list(set(df.columns) & set(features))]).loc[index, :]
        
        for feature in features:
            temp_list.append(list(total_df[feature]))
        data_list.append(temp_list)

    print(f"Total Time : {time.time() - start:.5f} sec")
    
    final_tensor = np.array(data_list)

    np.save(pos + file_name, final_tensor)


def ext_pedo(isval, pos):
    if(isval):
        file_path = '../data/val_dataset/ch2024_val__w_pedo.parquet.gzip'
        file_name = 'val_pedo.npy'
    else:
        file_path = '../data/test_dataset/ch2024_test_w_pedo.parquet.gzip'
        file_name = 'test_pedo.npy'
    df = pd.read_parquet(file_path)

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['date'] = df['timestamp'].dt.date
    df['time'] = df['timestamp'].dt.time

    grouped = df.groupby(['subject_id', 'date'])

    processed_data = []

    full_time_range = pd.date_range("00:00", "23:59", freq="T").time

    for (subject_id, date), group in grouped:
        group = group.set_index('time').reindex(full_time_range, fill_value=0).reset_index()
        
        burned_calories = group['burned_calories'].values
        distance = group['distance'].values
        running_steps = group['running_steps'].values
        speed = group['speed'].values
        steps = group['steps'].values
        step_frequency = group['step_frequency'].values
        walking_steps = group['walking_steps'].values
        
        feature_matrix = np.vstack([
            burned_calories,
            distance,
            running_steps,
            speed,
            steps,
            step_frequency,
            walking_steps
        ])
        
        processed_data.append(feature_matrix)

    final_array = np.stack(processed_data)

    final_tensor = torch.tensor(final_array, dtype=torch.float32)
    final_tensor = np.array(final_tensor)

    np.save(pos + file_name, final_tensor)


def ext_light(isval, pos):
    feature = ['m_light', 'w_light']
    if(isval):
        file_path = ['../data/val_dataset/ch2024_val__m_light.parquet.gzip', '../data/val_dataset/ch2024_val__w_light.parquet.gzip']
        file_name = ['val_m_light.npy', 'val_w_light.npy']
    else:
        file_path = ['../data/test_dataset/ch2024_test_m_light.parquet.gzip', '../data/test_dataset/ch2024_test_w_light.parquet.gzip']
        file_name = ['test_m_light.npy', 'test_w_light.npy']
    for k, file in enumerate(file_path):
        df = pd.read_parquet(file)

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        ignore_times = ['23:55:00', '23:56:00', '23:57:00', '23:58:00', '23:59:00']
        df = df[~df['timestamp'].dt.strftime('%H:%M:%S').isin(ignore_times)]

        df['timestamp'] = df['timestamp'].dt.round('10min')
        df['date'] = df['timestamp'].dt.date
        df['time'] = df['timestamp'].dt.time

        df[feature[k]] = df[feature[k]].astype(float)

        if(isval):
            ignore_subject_id = 3
            ignore_date = [pd.to_datetime('2023-10-06').date(), pd.to_datetime('2023-09-23').date(), pd.to_datetime('2023-09-21').date(), pd.to_datetime('2023-09-16').date(), 
                        pd.to_datetime('2023-09-09').date()]
            for i in range(len(ignore_date)):
                df = df[~((df['subject_id'] == ignore_subject_id) & (df['date'] == ignore_date[i]))]
            ignore_subject_id = 2
            ignore_date = pd.to_datetime('2023-10-01').date()
            df = df[~((df['subject_id'] == ignore_subject_id) & (df['date'] == ignore_date))]
        else:
            ignore_subject_id = 8
            ignore_date = [pd.to_datetime('2023-10-12').date(), pd.to_datetime('2023-10-14').date(), pd.to_datetime('2023-10-18').date(), pd.to_datetime('2023-10-21').date(),
                        pd.to_datetime('2023-11-10').date(),]
            for i in range(len(ignore_date)):
                df = df[~((df['subject_id'] == ignore_subject_id) & (df['date'] == ignore_date[i]))]
            ignore_subject_id = 6
            ignore_date = pd.to_datetime('2023-11-08').date()
            df = df[~((df['subject_id'] == ignore_subject_id) & (df['date'] == ignore_date))]

        df = df.groupby(['subject_id', 'date', 'time']).mean().reset_index()

        grouped = df.groupby(['subject_id', 'date'])

        processed_data = []

        full_time_range = pd.date_range("00:00", "23:50", freq="10T").time

        for (subject_id, date), group in grouped:
            group = group.set_index('time')
            
            group = group.reindex(full_time_range, fill_value=0)
            
            m_light = group[feature[k]].values
            
            processed_data.append(m_light)

        final_array = np.stack(processed_data)

        final_tensor = torch.tensor(final_array, dtype=torch.float32)
        final_tensor = np.array(final_tensor.unsqueeze(1))

        np.save(pos + file_name[k], final_tensor)


def ext_activity(isval, pos):
    feature = 'm_activity'
    if isval:
        file_path = '../data/val_dataset/ch2024_val__m_activity.parquet.gzip'
        file_name = 'val_activity.npy'
    else:
        file_path = '../data/test_dataset/ch2024_test_m_activity.parquet.gzip'
        file_name = 'test_activity.npy'

    df = pd.read_parquet(file_path)

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = df[df['timestamp'].dt.strftime('%H:%M:%S') <= '23:59:00']

    df['timestamp'] = df['timestamp'].dt.round('T')
    df['date'] = df['timestamp'].dt.date
    df['time'] = df['timestamp'].dt.time

    df[feature] = df[feature].astype(int)

    df = df.groupby(['subject_id', 'date', 'time']).mean().reset_index()

    grouped = df.groupby(['subject_id', 'date'])

    processed_data = []

    full_time_range = pd.date_range("00:00", "23:59", freq="T").time

    for (subject_id, date), group in grouped:
        group = group.set_index('time')
        
        group = group.reindex(full_time_range, fill_value=0)
        
        m_activity = group[feature].values
        
        processed_data.append(m_activity)

    final_array = np.stack(processed_data)

    final_tensor = torch.tensor(final_array, dtype=torch.int32)
    final_tensor = np.array(final_tensor.unsqueeze(1))

    np.save(pos + file_name, final_tensor)


def ext_ambience(isval, pos):
    feature = 'ambience_labels'
    if isval:
        file_path = '../data/val_dataset/ch2024_val__m_ambience.parquet.gzip'
        file_name = 'val_ambience.npy'
    else:
        file_path = '../data/test_dataset/ch2024_test_m_ambience.parquet.gzip'
        file_name = 'test_ambience.npy'

    df = pd.read_parquet(file_path)

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['timestamp'] = df['timestamp'].dt.round('2T')
    df['date'] = df['timestamp'].dt.date
    df['time'] = df['timestamp'].dt.time

    grouped = df.groupby(['subject_id', 'date'])

    processed_data = []

    full_time_range = pd.date_range("00:00", "23:58", freq="2T").time

    for (subject_id, date), group in grouped:
        group = group.set_index('time')
        
        group = group.reindex(full_time_range).reset_index()

        group[feature] = group[feature].apply(lambda x: [] if isinstance(x, float) and np.isnan(x) else x)

        ambience_labels = group[feature].values
        
        processed_data.append(ambience_labels)

    final_array = np.array(processed_data, dtype=object)[:, np.newaxis, :]

    np.save(pos + file_name, final_array)


def ext_usage_stats(isval, pos):
    feature = 'm_usage_stats'
    if isval:
        file_path = '../data/val_dataset/ch2024_val__m_usage_stats.parquet.gzip'
        file_name = 'val_usage_stats.npy'
    else:
        file_path = '../data/test_dataset/ch2024_test_m_usage_stats.parquet.gzip'
        file_name = 'test_usage_stats.npy'

    df = pd.read_parquet(file_path)

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['timestamp'] = df['timestamp'].dt.round('10T')
    df['date'] = df['timestamp'].dt.date
    df['time'] = df['timestamp'].dt.time

    grouped = df.groupby(['subject_id', 'date'])

    processed_data = []

    full_time_range = pd.date_range("00:00", "23:50", freq="10T").time

    for (subject_id, date), group in grouped:
        group = group.set_index('time')
        
        group = group.reindex(full_time_range).reset_index()

        group[feature] = group[feature].apply(lambda x: [] if isinstance(x, float) and np.isnan(x) else x)

        m_usage_stats = group[feature].values
        
        processed_data.append(m_usage_stats)

    final_array = np.array(processed_data, dtype=object)[:, np.newaxis, :]

    np.save(pos + file_name, final_array)
