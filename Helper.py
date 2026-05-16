import pandas as pd

def read_small_dataset():
	with open('UCISmallDataSet.txt', encoding='utf-8') as file:
		return file.read().splitlines()

def read_large_dataset_label_text_only():
	df = pd.read_csv('MendeleyLargeDataSet.csv', encoding='utf-8', usecols=['LABEL', 'TEXT'])
	data = df['LABEL'].str.lower() + ' ' + df['TEXT']
	return data.tolist()

#TODO: Simplify
def read_large_dataset_as_tab_lines(n_spam=500, n_ham=500):
	df = pd.read_csv('MendeleyLargeDataSet.csv', encoding='utf-8', usecols=['LABEL', 'TEXT'])
	df['LABEL'] = df['LABEL'].str.lower()
	df.loc[df['LABEL'] == 'smishing', 'LABEL'] = 'spam'

	ham_df = df[df['LABEL'] == 'ham'].head(n_ham)
	spam_df = df[df['LABEL'] == 'spam'].head(n_spam)

	combined = pd.concat([ham_df, spam_df], ignore_index=True)
	lines = (combined['LABEL'] + '\t' + combined['TEXT']).tolist()
	return lines
