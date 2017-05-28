import numpy as np
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizer import tokenize


def obtain_word_vector(w2v, word):
    try:
        return w2v[word]
    except Exception:
        return np.zeros(w2v.syn0.shape[1])


def obtain_feature_vector(w2v, document):
    """
    Returns a single vector representing the document supplied
    """
    tokens = tokenize(document)
    feature_vector = np.zeros(w2v.syn0.shape[1], dtype="float32")

    if len(tokens) == 0:
        return feature_vector

    for token in tokens:
        word_vector = obtain_word_vector(w2v, token)
        feature_vector = np.add(feature_vector, word_vector)
    # Average out the word vector
    # Loses contextual clues
    feature_vector = np.divide(feature_vector, len(tokens))

    return feature_vector


def obtain_feature_matrix(w2v, documents):
    input_matrix = np.zeros(
        (len(documents), w2v.syn0.shape[1]),
        dtype="float32"
    )

    for i, document in enumerate(documents):
        input_matrix[i] = obtain_feature_vector(w2v, document)

    return input_matrix


def get_vectorized_data(text_input, labels, use_word_2_vec=True):
    y = np.asarray(labels).astype(float)

    if use_word_2_vec:
        print("Building Word2Vec vectorizer...")
        print("Tokenizing...")
        document_tokens = [tokenize(document) for document in text_input]
        print("Tokenized...")
        w2v = word2vec.Word2Vec(
            document_tokens,
            sg=1,
            size=300,
            window=10
        )
        w2v.init_sims(replace=True)
        X = obtain_feature_matrix(w2v, text_input)
        print("Vectorizer Built...", end='\n\n')
        return X, y

    print("Building Tfidf vectorizer...")
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    tfs = tfidf.fit_transform(text_input)
    X = tfs.toarray()
    print("Vectorizer Built...", end='\n\n')
    return X, y
