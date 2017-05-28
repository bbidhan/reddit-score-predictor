import os
import pickle

from sklearn.utils import shuffle
from sklearn.svm import SVR

from regressor.helpers import sse


class SVMRegressor():
    """
    SVM based regressor
    """

    def __init__(self, vectorizer):
        # Base path
        self.base_path = os.path.dirname(__file__)

        # Path for pre-trained regressor and labels
        self.rgr_path = os.path.join(self.base_path, "svm.pkl")

        # Assign vectorizer to class
        self.vectorizer = vectorizer

        # Regressor to use
        self.regressor = None

        # Initialize the best params
        self.c = 3.3e4
        self.kernel = 'linear'
        self.gamma = 0.1

        # Test data size
        self.test_data_size = 0.33

        self.cross_validation_folds = 5

    def train(self, documents, labels):
        """
        Train regressor
        """

        if len(documents) != len(labels):
            raise Exception("No. of documents doesn't match the no. of labels")

        # Obtain corpus data
        documents, labels = shuffle(documents, labels)

        self.input_matrix = self.vectorizer.get_feature_matrix(documents)

        # Train SVM
        self.regressor = SVR(kernel=self.kernel, C=self.c, gamma=self.gamma)
        self.regressor.fit(self.input_matrix, labels)

        # Dump trained SVM
        pickle.dump(self.regressor, open(self.rgr_path, "wb"))

    def load_rgr(self):
        """ Load the pre-trained regressor """

        if (not(os.path.exists(self.rgr_path))):
            raise Exception("Pre trained regressor not found")

        self.regressor = pickle.load(open(self.rgr_path, "rb"))

    def predict(self, document):
        """
        Predict the class of given text
        """

        # Check and load regressor data
        self.load_rgr()

        if (document == ""):
            raise Exception("Document Empty!")

        feature_matrix = self.vectorizer.get_feature_matrix(document)

        predicted_scores = self.regressor.predict(feature_matrix)
        return predicted_scores

    def predict_document(self, document):
        """
        Predict the score of a given text
        """

        # Check and load regressor data
        self.load_rgr()

        if (document == ""):
            raise Exception("Document Empty!")

        feature_vector = self.vectorizer.get_feature_vector(document)

        predicted_score = self.regressor.predict(feature_vector)[0]
        return int(predicted_score)

    def calculate_errors(self, X_train, X_test, y_train, y_test):
        print('Calculating sum of squared errors...')
        y_train_pred = self.predict(X_train)
        y_test_pred = self.predict(X_test)
        sse_train = sse(y_train_pred, y_train)
        sse_test = sse(y_test_pred, y_test)
        print("SSE Train = {}".format(sse_train))
        print("SSE Test = {}".format(sse_test))
