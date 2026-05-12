from Models.BagOfWords.BagOfWords import BagOfWords


def test_bag_of_words_vectorisation():
	test_text = "The cat jumped over the other cat"
	labelled_input = [f"spam {test_text}"]

	print("Input:", labelled_input)

	bag_of_words = BagOfWords(labelled_input)

	bag_of_words.preprocess()
	print("After preprocess:")
	print("texts:", bag_of_words.texts)
	print("vocab:", sorted(bag_of_words.vocab))

	bag_of_words.vectorise()
	print("After vectorise:")
	print("bow_vectors:", bag_of_words.bow_vectors)

	fit_transform_model = BagOfWords(labelled_input)
	fit_transform_result = fit_transform_model.fit_transform()
	print("After fit_transform:")
	print(fit_transform_result)

def main():
    test_bag_of_words_vectorisation()

if __name__ == "__main__":
	main()
