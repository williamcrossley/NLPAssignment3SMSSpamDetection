#Algorithm
# PREPROCESSING
# Step 1: Assume labels and text are already provided as (label, text) tuples by the dataset reader.
#   This class consumes that DTO directly for easy access.
# Step 2: Convert all text to lowercase to ensure uniformity. Punctuation will be included but requires
#   more complex tokenisation to seperate from the words if required. The idea is spam seems to include a lot
#   of espectially repeated punctuation, so it may be useful to keep it in.
# Step 3: Tokenise. Split words based on spaces. I planned originally to do punctiation too but ran out of time, as well as this is supposed to be a benchmark representing simple NLP techniques, so its kind of out of scope.
# BUILD VECTORS
# Step 4: Build a single vocabulary from all texts (both spam and ham), then vectorise every text against this
#   shared vocabulary. Binary presence is used rather than frequency at i,j, as it works better with the Bernoulli classifier.
# Step 5: Split the resulting vector matrix into spam and ham subsets based on each row's label.
#   Since both subsets share the same vocabulary and column ordering, no merging or realignment is needed.
# CLASSIFICATION
#   At a high level, for each word we compare how often it appears in spam vs ham messages,
#   then convert that into a spam leaning weight (log-odds style). If a word is more common in spam,
#   it gets a positive weight, if more common in ham it gets a negative weight.
#   Our implementation leaves out the negative evidence term of a traditional BNB classifier, as it didn't seem to improve performance and just added complexity.
# Step 6: For a new message, vectorise it against the same shared vocabulary.
#   This gives us a binary presence vector where each column lines up with the same word used in training.
# Step 7: Score the message by starting with the prior and adding weights for words that are present.
#   Positive final score means spam, negative means ham (with an optional threshold if we want to tune sensitivity).
#   Smoothing (alpha) is also applied so unseen/rare words do not produce zero probability issues.

# We will be ignoring character look alike issues (since they are common in spam detection avoidance) as the reduced charset of GSM7 (SMS Standard) doesn't really allow it.
#   eg. in UTF8, crylic І (d086) and latin I (49) look the same, and may cause issues in vectorisation.
#   In GSM7, the only lookalikes are upside down excalm (64) and i (which arent very close), and maybe i with an accent (7).
#   So its not really a problem worth solving. Otherwise we would have to have a conversion step to convert all lookalikes.

# Note: The embedding this makes is insanely sparse, as SMS messages are short, and often very varied in content. By design this makes BOW a very bad embedding for this task,
#   ballooning very quickly and not providing much useful information.
#   However, during testing its accuracy isn't actually all that terrible, double however, I think its due to the spam used in both datasets being more 'conventional'
#   As spam evolves over time, or new types of spam appear, this model will likely struggle.

# Note: Assumed DTO for data is an array of tuples with the format (label, text).

# Note: For the classifier, started with an alpha value of 1, as some alpha is needed due to the training set bias but I wasn't sure how much
#   I found at 1.0 its number of false positives was very high compared to false negatives, so I reduced it to 0.5 to reduce the
#   counteraction of the spam / ham imbalance in the training set, which SIGNIFICANTLY reduced the false positive rate, with only a minor increase to false negatives.
#   eg. In the benchmark (500 ham, 500 spam), with alpha 1.0 it was 113 fp and 6 fn, with alpha 0.5 it was 45 fp, 7 fn. Overall a large improvement.
#       I'm sure there is possible tweaking to be done with the threshold as well, and I could run a regression test to approach the optimal values,
#       but I the purpose of this is a benchmark, not a fully optimised solution, as well as your tolerance for fp / fn depends on application, I will leave it at that.

import numpy as np

class BagOfWordsWithClassifier:
	def __init__(self, data, alpha=0.5, threshold=0.0):
		self.data = data
		self.alpha = alpha
		self.threshold = threshold
		self.vocab_list = None
		self.vocab_index = None
		self.log_prior = None
		self.feature_log_odds = None

	def fit(self):
		bow = BagOfWords(self.data)
		spam_vectors, ham_vectors, self.vocab_list = bow.fit_transform()
		self.vocab_index = {word: idx for idx, word in enumerate(self.vocab_list)}
		self.log_prior, self.feature_log_odds = BernoulliSpamClassifier._compute_word_spam_weights(
			spam_vectors, ham_vectors, self.alpha
		)
		return self

	def predict(self, text):
		text_vector = np.zeros(len(self.vocab_list), dtype=int)
		for word in BagOfWords._tokenise(text.lower()):
			if word in self.vocab_index:
				text_vector[self.vocab_index[word]] = 1
		score = BernoulliSpamClassifier._compute_spam_score(text_vector, self.log_prior, self.feature_log_odds)
		label = "spam" if score > self.threshold else "ham"
		return score, label


class BagOfWords:
	def __init__(self, data):
		self.data = data

	def fit_transform(self):
		texts, labels, vocab_list = self._preprocess(self.data)
		vectors = self._vectorise(texts, vocab_list)

		spam_indices = [i for i, label in enumerate(labels) if label == "spam"]
		ham_indices = [i for i, label in enumerate(labels) if label == "ham"]

		spam_vectors = vectors[spam_indices]
		ham_vectors = vectors[ham_indices]

		return spam_vectors, ham_vectors, vocab_list

	@staticmethod
	def _preprocess(data):
		texts = []
		labels = []
		vocab = set()

		for label, text in data:
			normalised_text = text.lower()
			texts.append(normalised_text)
			labels.append(label.lower())
			vocab.update(BagOfWords._tokenise(normalised_text))

		vocab_list = np.array(sorted(vocab))
		return texts, labels, vocab_list

	@staticmethod
	def _tokenise(text):
		return text.split()

	@staticmethod
	def _vectorise(texts, vocab_list):
		vocab_index = {word: idx for idx, word in enumerate(vocab_list)}
		vectors = []

		for text in texts:
			vector = np.zeros(len(vocab_list), dtype=int)
			for word in BagOfWords._tokenise(text):
				if word in vocab_index:
					vector[vocab_index[word]] = 1
			vectors.append(vector)

		return np.array(vectors)

class BernoulliSpamClassifier:
	@staticmethod
	def score_text(text, spam_vectors, ham_vectors, vocab_list, alpha=0.5, threshold=0.0):
		log_prior, feature_log_odds = BernoulliSpamClassifier._compute_word_spam_weights(
			spam_vectors,
			ham_vectors,
			alpha
		)

		text_vector = BernoulliSpamClassifier._vectorise_text_to_shared_vocab(text, vocab_list)
		score = BernoulliSpamClassifier._compute_spam_score(text_vector, log_prior, feature_log_odds)

		if score > threshold:
			label = "spam"
		else:
			label = "ham"

		return score, label

	@staticmethod
	def _compute_word_spam_weights(spam_vectors, ham_vectors, alpha):
		spam_document_count = len(spam_vectors)
		ham_document_count = len(ham_vectors)

		if spam_document_count == 0 or ham_document_count == 0:
			raise ValueError("Both spam and ham vectors are required to score text.")
		if alpha <= 0:
			raise ValueError("Alpha must be greater than 0 for smoothing.")

		spam_presence = np.sum(spam_vectors, axis=0)
		ham_presence = np.sum(ham_vectors, axis=0)

		spam_probability = (spam_presence + alpha) / (spam_document_count + 2 * alpha)
		ham_probability = (ham_presence + alpha) / (ham_document_count + 2 * alpha)

		log_prior = np.log(spam_document_count / ham_document_count)
		feature_log_odds = np.log(spam_probability / ham_probability)

		return log_prior, feature_log_odds

	@staticmethod
	def _vectorise_text_to_shared_vocab(text, vocab_list):
		vocab_index = {}
		text_vector = np.zeros(len(vocab_list), dtype=int)

		for index, word in enumerate(vocab_list):
			vocab_index[word] = index

		for word in BagOfWords._tokenise(text.lower()):
			if word in vocab_index:
				text_vector[vocab_index[word]] = 1

		return text_vector

	@staticmethod
	def _compute_spam_score(vector, log_prior, feature_log_odds):
		# Could use dot product of vector and feature_log_odds, since vector is binary presence,
		# but in case we ever want to make it frequency based I wont do that optimisation here.
		score = log_prior

		for index, is_present in enumerate(vector):
			if is_present:
				score += feature_log_odds[index]

		return score
