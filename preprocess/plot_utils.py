import pandas as pd
import matplotlib.pyplot as plt


def plot_label_dist(input_label):
    fig = plt.figure(figsize = (12,5))
    
    label = ["avg_up", "avg_down"]
    index = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
    pos = [241, 242, 243, 245, 246, 247, 248]
    title = ["Sleep Quality", "Emotion", "Stress", "Sleep Time", "Sleep Efficiency", "Sleep Latency", "WASO"]
    
    for i in range(7):
        ratio = [int(input_label[index[i]].sum()/len(input_label)*100), int((len(input_label)-input_label[index[i]].sum())/len(input_label)*100)]
        ax1 = fig.add_subplot(pos[i])
        ax1.pie(ratio, labels = label, autopct='%.1f%%')
        ax1.set_title(title[i])


def plot_val_test(df, subject_id, date):
    df_filtered = df[(df['subject_id'] == subject_id) & (df['timestamp'].dt.date == pd.to_datetime(date).date())]
    
    if df_filtered.empty:
        print("No data available for the given subject_id and date.")
        return
    
    features = df_filtered.columns[2:]
    num_features = len(features)
    
    fig, axes = plt.subplots(num_features, 1, figsize=(10, 3 * num_features))
    
    if num_features == 1:
        axes = [axes] 
    
    start_time = pd.Timestamp(date)
    end_time = start_time + pd.Timedelta(days=1)
    
    for i, feature in enumerate(features):
        axes[i].scatter(df_filtered['timestamp'], df_filtered[feature], s = 2, marker = '_', alpha = 1)
        axes[i].set_title(feature)
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel(feature)
        axes[i].set_xlim(start_time, end_time)
    
    plt.tight_layout()
    plt.show()