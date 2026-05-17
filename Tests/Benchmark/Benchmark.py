import sys
import os
from pathlib import Path

# Ensure the repo root is on sys.path for imports
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

# Flush progress output immediately so long training runs still show useful feedback in the terminal.
sys.stdout.reconfigure(line_buffering=True)
print("Starting...")

from tabulate import tabulate

from Helper import read_small_dataset, read_large_dataset_as_tab_lines, print_metrics
from Models.BagOfWords.BagOfWords import BagOfWords, BernoulliSpamClassifier
from Models.BERT.BERT import BERTSpamClassifier


def train_and_evaluate_bow(train_lines, test_lines):
	bow = BagOfWords(train_lines)
	spam_vectors, ham_vectors, vocab_list = bow.fit_transform()

	# Pre compute the spam-vs-ham weights once, then reuse them for every external benchmark message.
	log_prior, feature_log_odds = BernoulliSpamClassifier._compute_word_spam_weights(
		spam_vectors, ham_vectors, alpha=0.5
	)

	true_labels = []
	predicted_labels = []

	for label, text in test_lines:
		text_vector = BernoulliSpamClassifier._vectorise_text_to_shared_vocab(text, vocab_list)
		score = BernoulliSpamClassifier._compute_spam_score(text_vector, log_prior, feature_log_odds)
		prediction = "spam" if score > 0.0 else "ham"

		true_labels.append(label)
		predicted_labels.append(prediction)

	return true_labels, predicted_labels


def train_and_evaluate_bert(train_lines, test_lines, force_retrain=False):
	classifier = BERTSpamClassifier(
		dataset_lines=train_lines,
		output_dir=str(repo_root / "Models" / "BERT" / "artifacts" / "bert_benchmark"),
	)
	trainingResult = classifier.train(force_retrain=force_retrain)
	classifier.print_training_summary(trainingResult)

	true_labels = []
	predicted_labels = []

	for label, text in test_lines:
		prediction = classifier.predict(text, train_if_needed=False)
		true_labels.append(label)
		predicted_labels.append(prediction["label"])

	return true_labels, predicted_labels


def main():
	print("Loading datasets...")
	test_size = 500
	train_lines = read_small_dataset()
	test_lines = read_large_dataset_as_tab_lines(n_spam=test_size, n_ham=test_size)

	print(f"Training set: {len(train_lines)} samples (UCI)")
	print(f"Test set: {len(test_lines)} samples (Mendeley: {test_size} ham + {test_size} spam)")

	# Both models are trained on UCI and then evaluated on a separate Mendeley sample for a cross-dataset benchmark.
	# That makes this a harder and more realistic test than re-evaluating on another UCI split.

	# BagOfWords / Bernoulli
	print("\n[1/2] Training and evaluating Bag of Words (Bernoulli Naive Bayes)...")
	bow_all_labels, bow_pred_labels = train_and_evaluate_bow(train_lines, test_lines)
	bow_metrics = print_metrics("Bag of Words (Bernoulli NB)", bow_all_labels, bow_pred_labels)

	# BERT
	print("\n[2/2] Training and evaluating BERT...")
	bert_all_labels, bert_pred_labels = train_and_evaluate_bert(train_lines, test_lines)
	bert_metrics = print_metrics("BERT (bert-tiny fine-tuned)", bert_all_labels, bert_pred_labels)

	# Comparison
	print(f"\n{'='*60}")
	print(" Model Comparison Summary")
	print(f"{'='*60}")
	comparison_table = [
		["Bag of Words (Bernoulli NB)", f"{bow_metrics['accuracy']:.4f}", f"{bow_metrics['precision']:.4f}", f"{bow_metrics['recall']:.4f}", f"{bow_metrics['f1']:.4f}"],
		["BERT (bert-tiny)", f"{bert_metrics['accuracy']:.4f}", f"{bert_metrics['precision']:.4f}", f"{bert_metrics['recall']:.4f}", f"{bert_metrics['f1']:.4f}"],
	]
	print(tabulate(comparison_table, headers=["Model", "Accuracy", "Precision", "Recall", "F1"], tablefmt="github"))


if __name__ == "__main__":
	main()
