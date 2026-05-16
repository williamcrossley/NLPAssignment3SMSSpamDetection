import sys
import os
from pathlib import Path

# Ensure the repo root is on sys.path for imports
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
print("Starting...")

from tabulate import tabulate

from Helper import read_small_dataset, read_large_dataset_as_tab_lines, print_metrics
from Models.BagOfWords.BagOfWords import BagOfWords, BernoulliSpamClassifier
from Models.BERT.BERT import BERTSpamClassifier


def train_and_evaluate_bow(train_lines, test_lines):
	"Train BagOfWords/Bernoulli classifier on train_lines, evaluate on test_lines."
	bow = BagOfWords(train_lines)
	spam_vectors, spam_vocab = bow.fit_transform("spam")
	ham_vectors, ham_vocab = bow.fit_transform("ham")

	# Pre-compute shared vocab and aligned vectors once (expensive operations)
	merged_vocab_list = BernoulliSpamClassifier._merge_vocabularies(spam_vocab, ham_vocab)
	aligned_spam_vectors = BernoulliSpamClassifier._align_vectors_to_shared_vocab(
		spam_vectors, spam_vocab, merged_vocab_list
	)
	aligned_ham_vectors = BernoulliSpamClassifier._align_vectors_to_shared_vocab(
		ham_vectors, ham_vocab, merged_vocab_list
	)
	log_prior, feature_log_odds = BernoulliSpamClassifier._compute_word_spam_weights(
		aligned_spam_vectors, aligned_ham_vectors, alpha=1.0
	)

	true_labels = []
	predicted_labels = []

	for line in test_lines:
		label, text = line.split('\t', 1)
		label = label.strip().lower()

		text_vector = BernoulliSpamClassifier._vectorise_text_to_shared_vocab(text, merged_vocab_list)
		score = BernoulliSpamClassifier._compute_spam_score(text_vector, log_prior, feature_log_odds)
		prediction = "spam" if score > 0.0 else "ham"

		true_labels.append(label)
		predicted_labels.append(prediction)

	return true_labels, predicted_labels


def train_and_evaluate_bert(train_lines, test_lines, force_retrain=False):
	classifier = BERTSpamClassifier(
		dataset_lines=train_lines,
		output_dir=str(repo_root / "Models" / "BERT" / "artifacts" / "bert_benchmark"),
		test_size=0.1,
	)
	trainingResult = classifier.train(force_retrain=force_retrain)
	classifier.print_training_summary(trainingResult)

	true_labels = []
	predicted_labels = []

	for line in test_lines:
		label, text = line.split('\t', 1)
		label = label.strip().lower()
		prediction = classifier.predict(text, train_if_needed=False)
		true_labels.append(label)
		predicted_labels.append(prediction["label"])

	return true_labels, predicted_labels


def main():
	print("Loading datasets...")
	train_lines = read_small_dataset()
	test_lines = read_large_dataset_as_tab_lines(n_spam=500, n_ham=500)

	print(f"Training set: {len(train_lines)} samples (UCI)")
	print(f"Test set: {len(test_lines)} samples (Mendeley: 500 ham + 500 spam)")

	# BagOfWords / Bernoulli
	print("\n[1/2] Training and evaluating Bag of Words (Bernoulli Naive Bayes)...")
	bow_true, bow_pred = train_and_evaluate_bow(train_lines, test_lines)
	bow_metrics = print_metrics("Bag of Words (Bernoulli NB)", bow_true, bow_pred)

	# BERT
	print("\n[2/2] Training and evaluating BERT...")
	bert_true, bert_pred = train_and_evaluate_bert(train_lines, test_lines)
	bert_metrics = print_metrics("BERT (bert-tiny fine-tuned)", bert_true, bert_pred)

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
