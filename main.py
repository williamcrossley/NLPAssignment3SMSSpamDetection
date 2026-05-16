if __name__ == "__main__":
	print("Starting...")

from Models.BagOfWords.BagOfWords import BagOfWords, BernoulliSpamClassifier
from Models.BERT.BERT import BERTSpamClassifier
from Helper import read_small_dataset, read_large_dataset_label_text_only, print_single_text_test_results

def test_bow(text):
	print("Testing BagOfWords/Bernoulli Classifier on small dataset...")
	data = read_small_dataset()
	bow = BagOfWords(data)

	spam_vectors, spam_vocab = bow.fit_transform("spam")
	ham_vectors, ham_vocab = bow.fit_transform("ham")

	score, label = BernoulliSpamClassifier.score_text(text, spam_vectors, spam_vocab, ham_vectors, ham_vocab)

	print_single_text_test_results("BagOfWords/Bernoulli", score, label)

def test_bert(text):
	print("Testing BERT Classifier on small dataset...")
	classifier = BERTSpamClassifier()
	training_result = classifier.train()
	classifier.print_training_summary(training_result)

	prediction = classifier.predict(text)
	print_single_text_test_results("BERT", prediction["confidence"], prediction["label"])

	evaluation = classifier.evaluate()
	print("Dataset accuracy:", evaluation)


def main():
	text = "Congratulations! You've won a free ticket to the Bahamas. Reply 'WIN' to claim now!" # For some reason BERT thinks this is ham. TODO: INV
	print(f"\n\nTesting on text: {text}\n")
	test_bow(text)
	test_bert(text)

if __name__ == "__main__":
	main()
