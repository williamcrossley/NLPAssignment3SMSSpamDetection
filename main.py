from Models.BagOfWords.BagOfWords import BagOfWords
from Helper import read_small_dataset, read_large_dataset_label_text_only


def test_bow_vectorisation():
	test_text = "The cat jumped over the other cat"
	labelled_input = [f"spam {test_text}"]

	print("Input:", labelled_input)

	bag_of_words = BagOfWords(labelled_input)

	bag_of_words._preprocess()
	print("After preprocess:")
	print("texts:", bag_of_words.texts)
	print("vocab:", bag_of_words.vocab)

	bag_of_words._vectorise()
	print("After vectorise:")
	print("bow_vectors:", bag_of_words.bow_vectors)

	fit_transform_result = bag_of_words.fit_transform()
	print("After fit_transform:")
	print(fit_transform_result)

def run_bow_training():
    data = read_small_dataset()
    bow_model = BagOfWords(data)
    vectors, vocab_list = bow_model.fit_transform()
    print(vocab_list)
    print(vectors)
    bow_model.save_state()

def main():
    run_bow_training()

if __name__ == "__main__":
	main()
