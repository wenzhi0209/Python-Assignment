import re
import string

import nltk
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('mbti.csv')
#data.head()

# Text preprocessing steps - remove numbers, captial letters and punctuation


def alphanumeric(x): return re.sub(r"""\w*\d\w*""", ' ', x)


def punc_lower(x): return re.sub(
    '[%s]' % re.escape(string.punctuation), ' ', x.lower())


data['posts'] = data.posts.map(alphanumeric).map(punc_lower)
#data.head()

# split the data into feature and label
posts = data.posts  # inputs into model
type = data.type  # output of model

X_train, X_test, y_train, y_test = train_test_split(
    posts, type, test_size=0.3, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
model1.fit(X_train, y_train)
model1.score(X_test, y_test)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs', activation='logistic')
mlp.fit(X_train, y_train)
mlp.score(X_test, y_test)
