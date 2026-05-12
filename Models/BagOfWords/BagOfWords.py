#Algorithm
# PREPROCESSING
# Step 1: Process labeling by separating the first word as the label and the rest as the text.
#   Storing this in an array of tuples for easy access.
#   ONLY spam labels will be included, as ham labels are not needed for BOW.
# Step 2: Convert all text to lowercase to ensure uniformity. Punctuation will be included but requires
#   more complex tokenisation to seperate from the words if required. The idea is spam seems to include a lot
#   of espectially repeated punctuation, so it may be useful to keep it in.
# Step 3: Tokenise. Split words and punctuation.
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

import numpy as np

class BagOfWords:
    def __init__(self, data):
        self.data = data
        self.texts = []
        self.vocab = set()
        self.vocab_list = []
        self.bow_vectors = []
    
    def fit_transform(self):
        self._reset_state()
        self._preprocess()
        self._vectorise()
        return np.array(self.bow_vectors)

    def _preprocess(self):
        for line in self.data:
            label, text = line.split(' ', 1)
            if label == "spam":
                self.texts.append(text.lower())
                self.vocab.update(text.lower().split())

    def _vectorise(self):
        self.vocab_list = sorted(list(self.vocab))
        vocab_index = {word: idx for idx, word in enumerate(self.vocab_list)}
        
        for text in self.texts:
            vector = np.zeros(len(self.vocab_list), dtype=int)
            for word in text.split():
                if word in vocab_index:
                    vector[vocab_index[word]] += 1
            self.bow_vectors.append(vector)
        
    def _reset_state(self):
        self.texts = []
        self.vocab = set()
        self.vocab_list = []
        self.bow_vectors = []