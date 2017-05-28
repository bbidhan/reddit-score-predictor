from regressor import SVMRegressor, Word2VecVectorizer

domain = "en.wikipedia.org"
title = """
TIL a Russian child lived with a pack of dogs for two years
after he gave the dogs food.
In return the dog pack protected him and made him pack-leader.
He later relearned language and served in the Russian Army.
"""


def main():
    # Initialize the regressor
    rgr = SVMRegressor(Word2VecVectorizer())

    # Predicted score
    score = rgr.predict_document("{} {}".format(domain, title))

    print('The predicted score is: ', score)


if __name__ == '__main__':
    main()
