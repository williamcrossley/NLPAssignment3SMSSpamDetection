from ...Models.BagOfWords.BagOfWords import BagOfWords

def test_bow_vectorisation():
	test_text = "The cat jumped over the other cat"
	labelled_input = [f"spam\t{test_text}"]

	print("Input:", labelled_input)

	bag_of_words = BagOfWords(labelled_input)
	vectors, vocab_list = bag_of_words.fit_transform("spam")

	print("After fit_transform:")
	print("vocab:", vocab_list)
	print("bow_vectors:", vectors)