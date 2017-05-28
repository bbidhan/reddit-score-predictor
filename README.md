# Reddit Score Predictor

### Introduction
Predicts `score` of a reddit post, given it's `title` and `domain`.

### Dataset
[Reddit Top 2.5 Million](https://github.com/umbrae/reddit-top-2.5-million)

### Regression
* **SVM based Support Vector Regression**

### Vectorizer
* **Word2Vec**

### Installation
Download and extract the [latest release for your system](http://conda.pydata.org/miniconda.html).

For Linux 64-bit:
```sh
$ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh
$ conda create -n myenv python
$ source activate myenv
$ conda create -n myenv python
```
Install the dependencies

```sh
$ conda install numpy==1.11.1
$ conda install scikit-learn==0.17.1
$ conda install nltk==3.2.1
$ conda install gensim==0.12.4
$ conda install -c conda-forge matplotlib==1.5.2
```

Download corpus for nltk. In **python3**:
```python
import nltk
nltk.download("brown")      # Corpus
nltk.download("punkt")      # Tokenizer
nltk.download("stopwords")  # Stopwords
```

### Usage
Train the **word2vec** model and the **SVM Regressor**.
```sh
$ python3 train_w2v.py
$ python3 train_svm.py
```
Here's an example prediction (main.py):
```python
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

```
Just try running this command:
```sh
$ python3 main.py
```
### Notes, Codes and References
* Learned basics of text preprocessing(stemming, tokenizing, stop words removal). [Kaggle Part 1](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words)
* Learned about word vectors. [Kaggle Part 2](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-2-word-vectors)
    * Do not remove stop words because w2v relies on broader context of sentence
    * Large dataset = better
    * Could use a different dataset to train Word2Vec (For simplicity, I used the same subreddit dataset)
    * Played with different values of parameters
* Learned more about word vectors. [Kaggle Part 3](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-3-more-fun-with-word-vectors)
    * Get word vectors for each document and transform them into some feature matrix
    * Average out the word vectors so that we get a feature_vector for each document
    * Averaging loses contextual information and results in a model similar to Bag of Words model [Kaggle Part 4](https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-4-comparing-deep-and-non-deep-learning-methods)
    * But still a strong baseline for short text similarity tasks. [Quora](https://www.quora.com/How-do-I-compute-accurate-sentence-vectors-from-Word2Vec-tool)
* Learned how to use **Word2Vec** with gensim. [gensim](https://radimrehurek.com/gensim/models/word2vec.html)
* Used SVM with **linear kernel** because text data is linearly separable. [SVM Tutorial](http://www.svm-tutorial.com/2014/10/svm-linear-kernel-good-text-classification/)
* Could use deep neural nets for regression. [Regression with Deep Learning](http://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/)
