import pandas as pd

def read_small_dataset():
    f = open('UCISmallDataset.txt')
    data = f.readlines()
    f.close()
    return data

def read_large_dataset_label_text_only():
    df = pd.read_csv('UCILargeDataset.csv', encoding='utf-8', usecols=['label', 'text'])
    data = df['label'] + ' ' + df['text']
    return data.tolist()