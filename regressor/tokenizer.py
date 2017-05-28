import re
import nltk
from nltk.corpus import stopwords


def tokenize(text, remove_stopwords=False):
    letters_only = re.sub("[^a-zA-Z]", " ", text)
    # Convert to lower case
    lower_case = letters_only.lower()
    words = nltk.word_tokenize(lower_case)
    if remove_stopwords:
        # Remove stop words from "words"
        # Not recommended while using word2vec
        # Because it relies on the broader context of the sentence
        # In order to produce high-quality word vectors
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    return words
