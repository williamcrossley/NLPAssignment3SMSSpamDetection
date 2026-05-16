import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tabulate import tabulate

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

def print_metrics(model_name, true_labels, predicted_labels):
	label_names = ["ham", "spam"]
	true_binary = [1 if l == "spam" else 0 for l in true_labels]
	pred_binary = [1 if l == "spam" else 0 for l in predicted_labels]

	accuracy = accuracy_score(true_binary, pred_binary)
	precision = precision_score(true_binary, pred_binary)
	recall = recall_score(true_binary, pred_binary)
	f1 = f1_score(true_binary, pred_binary)
	cm = confusion_matrix(true_binary, pred_binary)

	print(f"\n{'='*60}")
	print(f" {model_name} Results")
	print(f"{'='*60}")

	metrics_table = [
		["Accuracy", f"{accuracy:.4f}"],
		["Precision", f"{precision:.4f}"],
		["Recall", f"{recall:.4f}"],
		["F1 Score", f"{f1:.4f}"],
	]
	print(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="github"))

	print(f"\nConfusion Matrix (rows=actual, cols=predicted):")
	cm_table = [
		["Ham", cm[0][0], cm[0][1]],
		["Spam", cm[1][0], cm[1][1]],
	]
	print(tabulate(cm_table, headers=["", "Pred Ham", "Pred Spam"], tablefmt="github"))

	print(f"\nClassification Report:")
	print(classification_report(true_binary, pred_binary, target_names=label_names))

	return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def print_single_text_test_results(model_name, confidence, label):
	print(f"\n{'='*60}")
	print(f" {model_name} Single Text Test Result")
	print(f"{'='*60}")
	print(f"Confidence: {confidence}")
	print(f"Predicted Label: {label}")
	print(f"\n\n")
