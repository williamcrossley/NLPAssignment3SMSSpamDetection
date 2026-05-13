#Algorithm
# PREPROCESSING
# Step 1: Process labeling by separating the first word as the label and the rest as the text.
#   Storing this in an array of tuples for easy access.
#   ONLY spam labels will be included, as ham labels are not needed for BOW.
# Step 2: Convert all text to lowercase to ensure uniformity. Punctuation will be included but requires
#   more complex tokenisation to seperate from the words if required. The idea is spam seems to include a lot
#   of espectially repeated punctuation, so it may be useful to keep it in.
# Step 3: Tokenise. Split words and punctuation. <- TODO: punctuation tokenisation.
# BUILD VOCAB
# Step 4: Vectorise. Essentially make a vector that counts the frequency of each word.
# TEST INSATNCE
# Step 5: For a given text, we can 

# Note: Character encodings will remain in UTF 8, as that is what the dataset was originally in. 
#   GSM7 is the standard for SMS, but its hard to work with in file based systems wanting full bytes.
#   This implementation will remain agnostic to the character encoding, so if required to be used in a
#   GSM7 environment, it will only require a change to the dataset character encoding.

# We will also be ignoring character look alike issues (since they are common in spam detection avoidance) as the reduced charset of GSM7 doesn't really allow it.
#   eg. in UTF8, crylic І (d086) and latin I (49) look the same, and may cause issues in vectorisation.
#   In GSM7, the only lookalikes are upside down excalm (64) and i (which arent very close), and maybe i with an accent (7).
#   So its not really a problem worth solving. Otherwise we would have to have a conversion step to convert all lookalikes.

# Note: The embedding this makes is insanely sparse, as SMS messages are short, and often very varied in content. By design this makes BOW a very bad embedding for this task,
#   ballooning very quickly and not providing much useful information.

import numpy as np

class BagOfWords:
    def __init__(self, data):
        self.data = data
    
    def fit_transform(self, label_filter="spam", binary=False):
        texts, vocab_list = self._preprocess(self.data, label_filter=label_filter)
        vectors = self._vectorise(texts, vocab_list, binary=binary)
        return vectors, vocab_list

    @staticmethod
    def save_state(vectors, vocab_list, file_name="./Models/BagOfWords/bag_of_words_state.txt"): #debug, probably wont include in final version unless required
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write("Vocabulary:\n")
            f.write("\t".join(map(str, vocab_list)))
            f.write("\n\nBag of Words Vectors:\n")
            for vector in vectors:
                f.write(" ".join(map(str, vector)))
                f.write("\n")
        return file_name

    @staticmethod
    def _preprocess(data, label_filter="spam"):
        allowed_labels = BagOfWords._normalise_label_filter(label_filter)
        texts = []
        vocab = set()

        for line in data:
            label, text = line.split('\t', 1)
            label = label.strip().lower()
            normalised_text = text.lower()

            if allowed_labels is None or label in allowed_labels:
                texts.append(normalised_text)
                vocab.update(normalised_text.split())

        vocab_list = np.array(sorted(vocab))
        return texts, vocab_list

    @staticmethod
    def _vectorise(texts, vocab_list, binary=False):
        vocab_index = {word: idx for idx, word in enumerate(vocab_list)}
        vectors = []

        for text in texts:
            vector = np.zeros(len(vocab_list), dtype=int)
            for word in text.split():
                if word in vocab_index:
                    if binary:
                        vector[vocab_index[word]] = 1
                    else:
                        vector[vocab_index[word]] += 1
            vectors.append(vector)

        return np.array(vectors)

    @staticmethod
    def _normalise_label_filter(label_filter):
        if label_filter is None:
            return None

        if isinstance(label_filter, str):
            return {label_filter.lower()}

        normalised_labels = set()

        for label in label_filter:
            normalised_labels.add(label.lower())

        return normalised_labels

# TODO: Add reference to report for algorithm source
class BernoulliSpamClassifier:
    @staticmethod
    def score_text(text, spam_vectors, spam_vocab_list, ham_vectors, ham_vocab_list, alpha=1.0, threshold=0.0):
        merged_vocab_list = BernoulliSpamClassifier._merge_vocabularies(spam_vocab_list, ham_vocab_list)

        aligned_spam_vectors = BernoulliSpamClassifier._align_vectors_to_shared_vocab(
            spam_vectors,
            spam_vocab_list,
            merged_vocab_list
        )
        aligned_ham_vectors = BernoulliSpamClassifier._align_vectors_to_shared_vocab(
            ham_vectors,
            ham_vocab_list,
            merged_vocab_list
        )

        log_prior, feature_log_odds = BernoulliSpamClassifier._compute_word_spam_weights(
            aligned_spam_vectors,
            aligned_ham_vectors,
            alpha
        )

        text_vector = BernoulliSpamClassifier._vectorise_text_to_shared_vocab(text, merged_vocab_list)
        score = BernoulliSpamClassifier._compute_spam_score(text_vector, log_prior, feature_log_odds)

        if score > threshold:
            label = "spam"
        else:
            label = "ham"

        return score, label

    @staticmethod
    def _merge_vocabularies(spam_vocab_list, ham_vocab_list):
        merged_vocab = set()

        for word in spam_vocab_list:
            merged_vocab.add(str(word))

        for word in ham_vocab_list:
            merged_vocab.add(str(word))

        return sorted(merged_vocab)

    @staticmethod
    def _align_vectors_to_shared_vocab(vectors, source_vocab_list, target_vocab_list):
        source_index = {}
        target_index = {}
        aligned_vectors = []

        for index, word in enumerate(source_vocab_list):
            source_index[str(word)] = index

        for index, word in enumerate(target_vocab_list):
            target_index[str(word)] = index

        for vector in vectors:
            aligned_vector = np.zeros(len(target_vocab_list), dtype=int)

            for word in source_index:
                source_position = source_index[word]
                target_position = target_index[word]

                if vector[source_position] > 0:
                    aligned_vector[target_position] = 1

            aligned_vectors.append(aligned_vector)

        return np.array(aligned_vectors)

    @staticmethod
    def _compute_word_spam_weights(spam_vectors, ham_vectors, alpha):
        spam_document_count = len(spam_vectors)
        ham_document_count = len(ham_vectors)

        if spam_document_count == 0 or ham_document_count == 0:
            raise ValueError("Both spam and ham vectors are required to score text.")

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

        for word in text.lower().split():
            if word in vocab_index:
                text_vector[vocab_index[word]] = 1

        return text_vector

    @staticmethod
    def _compute_spam_score(vector, log_prior, feature_log_odds):
        score = log_prior

        for index, is_present in enumerate(vector):
            if is_present:
                score += feature_log_odds[index]

        return score