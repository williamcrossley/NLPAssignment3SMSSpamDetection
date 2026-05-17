print("Starting...")

from Models.BagOfWords.BagOfWords import BagOfWordsWithClassifier
from Models.BERT.BERT import BERTSpamClassifier
from Helper import read_small_dataset, print_single_text_test_results

# Single-text checks are only sanity tests; the benchmark script is the main comparative evaluation.
def test_bow(spam_text, ham_text, dataset_lines):
	print("Testing BagOfWords/Bernoulli Classifier on small dataset...")
	classifier = BagOfWordsWithClassifier(dataset_lines)
	classifier.fit()

	spam_score, spam_label = classifier.predict(spam_text)
	print_single_text_test_results("BagOfWords/Bernoulli", spam_text, spam_score, spam_label)

	ham_score, ham_label = classifier.predict(ham_text)
	print_single_text_test_results("BagOfWords/Bernoulli", ham_text, ham_score, ham_label)

def test_bert(spam_text, ham_text, dataset_lines):
	print("Testing BERT Classifier on small dataset...")
	classifier = BERTSpamClassifier(dataset_lines=dataset_lines)
	training_result = classifier.train()
	classifier.print_training_summary(training_result)

	prediction = classifier.predict(spam_text)
	print_single_text_test_results("BERT", spam_text, prediction["confidence"], prediction["label"])

	prediction = classifier.predict(ham_text)
	print_single_text_test_results("BERT", ham_text, prediction["confidence"], prediction["label"])

	if training_result["post_eval_accuracy"] is not None:
		print("Dataset accuracy:", training_result["post_eval_accuracy"])


def main():
	spam_text = "dear paytm customer your paytm kyc ingest expired. contact customer care no-6299257179 immediately. your account will block within 24 hr. thank you paytm team."
	ham_text = "hey, are we still on for dinner tonight? let me know if you want to change the time or place. looking forward to it!"
	dataset_lines = read_small_dataset()
	test_bow(spam_text, ham_text, dataset_lines)
	test_bert(spam_text, ham_text, dataset_lines)

if __name__ == "__main__":
	main()
