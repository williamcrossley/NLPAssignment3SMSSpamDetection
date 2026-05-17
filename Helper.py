import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tabulate import tabulate

def read_small_dataset():
	with open('UCISmallDataSet.txt', encoding='utf-8') as file:
		return [_parse_tab_labelled_line(line) for line in file.read().splitlines()]

def read_large_dataset_as_tab_lines(n_spam=500, n_ham=500):
	ham_rows = []
	spam_rows = []

	with open('MendeleyLargeDataSet.csv', encoding='utf-8') as f:
		reader = csv.DictReader(f)
		for row in reader:
			if len(ham_rows) >= n_ham and len(spam_rows) >= n_spam:
				break

			label = _normalise_label(row['LABEL'])
			text = row['TEXT'].strip()

			if label == 'ham' and len(ham_rows) < n_ham:
				ham_rows.append((label, text))
			elif label == 'spam' and len(spam_rows) < n_spam:
				spam_rows.append((label, text))

	return ham_rows + spam_rows

def print_metrics(model_name, all_labels, predicted_labels):
	label_names = ["ham", "spam"]
	all_binary = [1 if l == "spam" else 0 for l in all_labels]
	pred_binary = [1 if l == "spam" else 0 for l in predicted_labels]

	accuracy = accuracy_score(all_binary, pred_binary)
	precision = precision_score(all_binary, pred_binary)
	recall = recall_score(all_binary, pred_binary)
	f1 = f1_score(all_binary, pred_binary)
	cm = confusion_matrix(all_binary, pred_binary)

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
	print(classification_report(all_binary, pred_binary, target_names=label_names))

	return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def print_single_text_test_results(model_name, confidence, label):
	print(f"\n{'='*60}")
	print(f" {model_name} Single Text Test Result")
	print(f"{'='*60}")
	print(f"Confidence: {confidence}")
	print(f"Predicted Label: {label}")
	print(f"\n\n")

def _normalise_label(label):
	normalised_label = label.strip().lower()
	if normalised_label == 'smishing':
		normalised_label = 'spam'

	return normalised_label


def _parse_tab_labelled_line(line):
	label, text = line.split('\t', 1)
	return _normalise_label(label), text.strip()
