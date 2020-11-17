import pandas as pd


def csv_processing():
    file = pd.read_csv('Chest_xray_Corona_Metadata.csv')
    normal = file.loc[file['Label'] == 'Normal']
    virus = file.loc[(file['Label'] == 'Pnemonia') & (file['Label_1_Virus_category'] == 'Virus')]
    bacteria = file.loc[(file['Label'] == 'Pnemonia') & (file['Label_1_Virus_category'] == 'bacteria')]

    normal_train = normal.loc[normal['Dataset_type'] == 'TRAIN']
    virus_train = virus.loc[virus['Dataset_type'] == 'TRAIN']
    bacteria_train = bacteria.loc[bacteria['Dataset_type'] == 'TRAIN']

    normal_test = normal.loc[normal['Dataset_type'] == 'TEST']
    virus_test = virus.loc[virus['Dataset_type'] == 'TEST']
    bacteria_test = bacteria.loc[bacteria['Dataset_type'] == 'TEST']

    # normal_train.to_csv('normal_train.csv')
    # virus_train.to_csv('virus_train.csv')
    # bacteria_train.to_csv('bacteria_train.csv')

    # normal_test.to_csv('normal_test.csv')
    # virus_test.to_csv('virus_test.csv')
    # bacteria_test.to_csv('bacteria_test.csv')

    train = normal_train
    train = train.append(virus_train)
    train = train.append(bacteria_train)

    test = normal_test
    test = test.append(virus_test)
    test = test.append(bacteria_test)

    train.to_csv('processed_data/train.csv')
    test.to_csv('processed_data/test.csv')


if __name__ == '__main__':
    csv_processing()
