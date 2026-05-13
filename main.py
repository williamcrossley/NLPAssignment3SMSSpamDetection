from Models.BagOfWords.BagOfWords import BagOfWords, BernoulliSpamClassifier
from Helper import read_small_dataset, read_large_dataset_label_text_only

# def run_bow_training():
#     data = read_small_dataset()
#     bow_model = BagOfWords(data)
#     vectors, vocab_list = bow_model.fit_transform()
#     print(vocab_list)
#     print(vectors)
#     bow_model._save_state()

def test_bow():
    data = read_small_dataset()
    bow = BagOfWords(data)

    spam_vectors, spam_vocab = bow.fit_transform("spam", binary=True)
    ham_vectors, ham_vocab = bow.fit_transform("ham", binary=True)

    text = "Hey man how you doing? Wanna catch up later?"
    score, label = BernoulliSpamClassifier.score_text(text, spam_vectors, spam_vocab, ham_vectors, ham_vocab)

    print("Score:", score)
    print("Prediction:", label)

def main():
    test_bow()

if __name__ == "__main__":
	main()
