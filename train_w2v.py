from regressor import Word2VecVectorizer
from regressor.helpers import get_documents_and_labels


if __name__ == "__main__":
    # Load data
    documents, labels = get_documents_and_labels('todayilearned.csv')

    # Train vectorizer
    vectorizer = Word2VecVectorizer()
    vectorizer.train(documents)
    print("Done...")
