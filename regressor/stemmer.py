from nltk.stem.porter import PorterStemmer

from .tokenizer import tokenize


class EnglishStemmer():
    """ Stemmer for English words """

    def __init__(self):
        self.stemmer = PorterStemmer()

    def stem_text(self, text):
        """ Returns stemlist of the give text """

        tokens = tokenize(text)

        stemmed = [self.stemmer.stem(word) for word in tokens]
        return stemmed
