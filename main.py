print("Starting...")

from Models.BagOfWords.BagOfWords import BagOfWords, BernoulliSpamClassifier
from Models.BERT.BERT import BERTSpamClassifier
from Helper import read_small_dataset, print_single_text_test_results

def test_bow(text, dataset_lines):
	print("Testing BagOfWords/Bernoulli Classifier on small dataset...")
	bow = BagOfWords(dataset_lines)

	spam_vectors, spam_vocab = bow.fit_transform("spam")
	ham_vectors, ham_vocab = bow.fit_transform("ham")

	score, label = BernoulliSpamClassifier.score_text(text, spam_vectors, spam_vocab, ham_vectors, ham_vocab)

	print_single_text_test_results("BagOfWords/Bernoulli", score, label)

def test_bert(text, dataset_lines):
	print("Testing BERT Classifier on small dataset...")
	classifier = BERTSpamClassifier(dataset_lines=dataset_lines)
	training_result = classifier.train()
	classifier.print_training_summary(training_result)

	prediction = classifier.predict(text)
	print_single_text_test_results("BERT", prediction["confidence"], prediction["label"])

	if training_result["post_eval_accuracy"] is not None:
		print("Dataset accuracy:", training_result["post_eval_accuracy"])


def main():
	text = "Congratulations! You've won a free ticket to the Bahamas. Reply 'WIN' to claim now!" # For some reason BERT thinks this is ham. TODO: INV
	dataset_lines = read_small_dataset()
	print(f"\n\nTesting on text: {text}\n")
	test_bow(text, dataset_lines)
	test_bert(text, dataset_lines)

if __name__ == "__main__":
	main()
