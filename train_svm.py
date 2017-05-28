from sklearn.cross_validation import train_test_split
from regressor import Word2VecVectorizer, SVMRegressor
from regressor.helpers import get_documents_and_labels


if __name__ == "__main__":
    print('Loading dataset...')
    documents, labels = get_documents_and_labels('todayilearned.csv')

    # Initialize vectorizer
    vectorizer = Word2VecVectorizer()

    print('Randomly Splitting dataset into train and test set...')
    X_train, X_test, y_train, y_test = train_test_split(
        documents, labels, test_size=0.33, random_state=11)

    rgr = SVMRegressor(vectorizer)
    rgr.train(X_train, y_train)
    rgr.calculate_errors(X_train, X_test, y_train, y_test)

    print("Done...")
