import shutil
import pandas as pd
import os

train = pd.read_csv('processed_data/train.csv')
test = pd.read_csv('Cprocessed_data/test.csv')
train_img_path = 'Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train'
test_img_path = 'Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test'
train_path = 'processed_data/train'
test_path = 'processed_data/test'

for i in train.index:
    filename = train.loc[i].values[1]
    file = os.path.join(train_img_path, filename)
    print(file)
    if train.loc[i].values[6] == 0:
        shutil.move(file, os.path.join(train_path, str(0), filename))
    elif train.loc[i].values[6] == 1:
        shutil.move(file, os.path.join(train_path, str(1), filename))
    elif train.loc[i].values[6] == 2:
        shutil.move(file, os.path.join(train_path, str(2), filename))

for i in test.index:
    filename = test.loc[i].values[1]
    file = os.path.join(test_img_path, filename)
    if test.loc[i].values[6] == 0:
        shutil.move(file, os.path.join(test_path, str(0), filename))
    elif test.loc[i].values[6] == 1:
        shutil.move(file, os.path.join(test_path, str(1), filename))
    elif test.loc[i].values[6] == 2:
        shutil.move(file, os.path.join(test_path, str(2), filename))
