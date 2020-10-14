import os
import glob
import numpy as np
import pandas as pd

# splitter setting
train = 61; val = 24; test = 16

data_path = "../datas/"
save_path = "./UCF101_few_shot_labels/"

file_name = ["train.csv", "val.csv", "test.csv"]

if not os.path.exists(save_path):
    os.makedirs(save_path)

with open(os.path.join(data_path, "UCF101_labels/classInd.txt")) as f:
    categories = f.readlines()

categories = np.random.permutation(categories)

folder_name = []
for d in glob.glob(os.path.join(data_path, "UCF101_frames/*")):
    if os.path.isdir(d):
        folder_name.append(d.split("\\" if os.name == 'nt' else "/")[-1])
folder_name = pd.DataFrame(folder_name)

# save train labels
with open(os.path.join(save_path, file_name[0]), 'w') as f:
    first = True
    for i, c in enumerate(categories[:train]):
        print("writing {}===".format(c))
        lines = np.concatenate(folder_name.loc[folder_name[0].str.contains('_' + c.strip('\n').split(' ')[1] + '_')].values, axis=0)
        for line in lines:
            f.write(str(i+1) + ',' + line) if first else f.write('\n' + str(i+1) + ',' + line)
            first = False

# save val labels
with open(os.path.join(save_path, file_name[1]), 'w') as f:
    first = True
    for i, c in enumerate(categories[train:train+val]):
        print("writing {}===".format(c))
        lines = np.concatenate(folder_name.loc[folder_name[0].str.contains('_' + c.strip('\n').split(' ')[1] + '_')].values, axis=0)
        for line in lines:
            f.write(str(i+1) + ',' + line) if first else f.write('\n' + str(i+1) + ',' + line)
            first = False

# save test labels
with open(os.path.join(save_path, file_name[2]), 'w') as f:
    first = True
    for i, c in enumerate(categories[train+val:]):
        print("writing {}===".format(c))
        lines = np.concatenate(folder_name.loc[folder_name[0].str.contains('_' + c.strip('\n').split(' ')[1] + '_')].values, axis=0)
        for line in lines:
            f.write(str(i+1) + ',' + line) if first else f.write('\n' + str(i+1) + ',' + line)
            first = False