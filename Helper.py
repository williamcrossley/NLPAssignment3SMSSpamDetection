import pandas as pd

def read_small_dataset():
    with open('UCISmallDataSet.txt', encoding='utf-8') as file:
        return file.read().splitlines()

def read_large_dataset_label_text_only():
    df = pd.read_csv('MendeleyLargeDataSet.csv', encoding='utf-8', usecols=['LABEL', 'TEXT'])
    data = df['LABEL'].str.lower() + ' ' + df['TEXT']
    return data.tolist()