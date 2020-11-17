import pandas as pd


def read():
    train = pd.read_csv('processed_data/train.csv')
    test = pd.read_csv('processed_data/test.csv')
    filename_train = train['X_ray_image_name']
    filename_test = test['X_ray_image_name']
    label_train = []
    label_test = []
    ind_train = []
    ind_test = []
    for i in train.index:
        print(train.loc[i].values[3])
        if train.loc[i].values[2] == 'Normal':
            print(train.loc[i].values[3])
            ind_train.append(train.loc[i].values[0])
            label_train.append(0)
        elif train.loc[i].values[2] == 'Pnemonia' and train.loc[i].values[5] == 'Virus':
            ind_train.append(train.loc[i].values[0])
            label_train.append(1)
        elif train.loc[i].values[2] == 'Pnemonia' and train.loc[i].values[5] == 'bacteria':
            ind_train.append(train.loc[i].values[0])
            label_train.append(2)

    for i in test.index:
        if test.loc[i].values[3] == 'Normal':
            ind_test.append(test.loc[i].values[0])
            label_test.append(0)
        elif test.loc[i].values[3] == 'Pnemonia' and test.loc[i].values[6] == 'Virus':
            ind_test.append(test.loc[i].values[0])
            label_test.append(1)
        elif test.loc[i].values[3] == 'Pnemonia' and test.loc[i].values[6] == 'bacteria':
            ind_test.append(test.loc[i].values[0])
            label_test.append(2)

    label_train = pd.DataFrame({'index': ind_train,
                                'Label': label_train})
    label_test = pd.DataFrame({'index': ind_test,
                               'Label': label_test})

    label_train.to_csv('label_train.csv')
    label_test.to_csv('label_test.csv')


if __name__ == '__main__':
    read()
