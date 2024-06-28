import data_transform
import pandas as pd
import numpy as np

label_clf = pd.read_csv('../data/val_label.csv')

Q1 = np.array(label_clf['Q1'])
Q2 = np.array(label_clf['Q2'])
Q3 = np.array(label_clf['Q3'])
S1 = np.array(label_clf['S1'])
S2 = np.array(label_clf['S2'])
S3 = np.array(label_clf['S3'])
S4 = np.array(label_clf['S4'])

file_path = '../data_npy/val/'

np.save(file_path + 'Q1_label.npy', Q1)
np.save(file_path + 'Q2_label.npy', Q2)
np.save(file_path + 'Q3_label.npy', Q3)
np.save(file_path + 'S1_label.npy', S1)
np.save(file_path + 'S2_label.npy', S2)
np.save(file_path + 'S3_label.npy', S3)
np.save(file_path + 'S4_label.npy', S4)


data_transform.ext_acc(True, "../data_npy/val/")
data_transform.ext_acc(False, "../data_npy/test/")
data_transform.ext_gps_hr(True, "../data_npy/val/", 'gps')
data_transform.ext_gps_hr(False, "../data_npy/test/", 'gps')
data_transform.ext_gps_hr(True, "../data_npy/val/", 'hr')
data_transform.ext_gps_hr(False, "../data_npy/test/", 'hr')

data_transform.ext_light(True, "../data_npy/val/")
data_transform.ext_light(False, "../data_npy/test/")
data_transform.ext_pedo(True, "../data_npy/val/")
data_transform.ext_pedo(False, "../data_npy/test/")
data_transform.ext_activity(True, "../data_npy/val/")
data_transform.ext_activity(False, "../data_npy/test/")