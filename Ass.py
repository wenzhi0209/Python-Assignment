import re
import string
import nltk
from nltk.corpus import stopwords

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

def train_adam_model():
    start_time = time.time()

    mlp = MLPClassifier(solver='adam', activation='logistic', hidden_layer_sizes=(80), max_iter=1000)

    mlp.fit(X_train_tfidf, y_train)

    minutes=(time.time() - start_time)/60
    print("---Training Time %0.2f minutes ---" %minutes)
    print("---Hidden Layer 80 with solver=adam, activation=logistic, NLTK stopword---")
    print("")
    
    joblib.dump(filename='MLP_ADAM',value=mlp)
    modelLoad = joblib.load(filename="MLP_ADAM")
    show_result()
    


def train_lbfgs_model():
    start_time = time.time()

    mlp = MLPClassifier(solver='lbfgs', activation='logistic', hidden_layer_sizes=(80), max_iter=1000)

    mlp.fit(X_train_tfidf, y_train)

    minutes=(time.time() - start_time)/60
    print("---Training Time %0.2f minutes ---" %minutes)
    print("---Hidden Layer 80 with solver=lbfgs, activation=logistic, NLTK stopword---")
    print("")
    
    joblib.dump(filename='MLP_LBFGS',value=mlp)
    modelLoad = joblib.load(filename="MLP_LBFGS")
    show_result()

def show_result():
    print("TEST SET")
    print("Accuracy: ", modelLoad.score(X_test_tfidf, y_test))
    #print("Confusion Matrix:")
    #print(confusion_matrix(y_test, modelLoad.predict(X_test_tfidf)))
    print("Classification Report:")
    print(classification_report(y_test, modelLoad.predict(X_test_tfidf)))

    cm_TestSet = np.array(confusion_matrix(y_test, modelLoad.predict(X_test_tfidf)))
    cm_TestSetDF = pd.DataFrame(cm_TestSet, index=['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP',
                                        'ESTJ', 'ESTP', 'INFJ', 'INFP', 'INTJ', 'INTP',
                                        'ISFJ', 'ISFP', 'ISTJ', 'ISTP'], 
                                columns=['predict_ENFJ','predict_ENFP','predict_ENTJ',
                                        'predict_ENTP','predict_ESFJ','predict_ESFP',
                                        'predict_ESTJ','predict_ESTP','predict_INFJ',
                                        'predict_INFP','predict_INTJ','predict_INTP',
                                        'predict_ISFJ','predict_ISFP','predict_ISTJ',
                                        'predict_ISTP'])

    fig, ax = plt.subplots(figsize=(14,10)) 
    plt.title('Confusion Matrix for MLP Classifier Test Set', fontsize=16,
            fontweight='bold', y=1.02)
    sns.heatmap(cm_TestSetDF, robust=True, annot=True, linewidth=0.5, 
                fmt='', cmap='RdBu_r', vmax=303, ax=ax)
    plt.xticks(fontsize=12)
    plt.yticks(rotation=0, fontsize=12)



print("Please wait data laoding...")
data = pd.read_csv('mbti.csv')

# Text preprocessing steps - remove numbers, captial letters and punctuation
# remove url
def alphanumeric(x): return re.sub(r"""\w*\d\w*""", ' ', x)
def punc_lower(x): return re.sub('[%s]' % re.escape(string.punctuation),' ', x.lower())
def urlLink(x): return re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b',' ', x)

data['posts'] = data.posts.map(urlLink).map(alphanumeric).map(punc_lower)


# split the data into feature and label
posts = data.posts  # inputs into model
types = data.type  # output of model

#random_state=42
X_train, X_test, y_train, y_test = train_test_split(
    posts, types, test_size=0.2,random_state=42)


#apply nltk stopword
#stopwords.words('english')

#apply default stopword
#stop_words='english'

tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


uInput=4
while uInput>3:
    print("1.Start the new training.")
    print("2.Load the trained model.")
    print("3.Exit")
    print("")

    uInput=int(input())
    if(uInput==1):
        uInput2=4
        while uInput2>3:
            print("Select the model you want to train:")
            print("1.Train Adam solver model")
            print("2.Train Lbfgs solver model.")
            print("3.Exit")
            print("")
            uInput2=int(input())
            if(uInput2==1):
                print("Adam solver model training...")
                train_adam_model()
            elif(uInput2==2):
                print("Lbfgs solver model training...")
                train_lbfgs_model()
            elif(uInput2==3):
                print("Return to module menu...")
                uInput=4
            else:
                print("Invalid Input...")
                uInput2=4
        
    elif(uInput==2):
        uInput2=4
        while uInput2>3:
            print("Select the model you want to load:")
            print("1.Load Adam solver model")
            print("2.Load Lbfgs solver model.")
            print("3.Exit")
            print("")
            uInput2=int(input())
            if(uInput2==1):
                print("Adam solver model loading...")
                modelLoad = joblib.load(filename="MLP_ADAM")
                show_result()
            elif(uInput2==2):
                print("Lbfgs solver model loading...")
                modelLoad = joblib.load(filename="MLP_LBFGS")
                show_result()
            elif(uInput2==3):
                print("Return to module menu...")
                uInput=4
            else:
                print("Invalid Input...")
                uInput2=4
        
    elif(uInput==3):
        print("Module existing...")
        break
    elif(uInput==4):
        uInput=5
    else:
        print("Invalid Input...")
        uInput=5







#print("TRAINING SET")
#print("Accuracy: ", modelLoad.score(X_train_tfidf, y_train))
#print("Confusion Matrix:")
#print(confusion_matrix(y_train, modelLoad.predict(X_train_tfidf)))
#print("Classification Report:")
#print(classification_report(y_train, modelLoad.predict(X_train_tfidf)))



#strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True)
#score=mlp.score(X_test_tfidf, y_test)
#precision= np.mean(cross_val_score(MLPClassifier(), X_test_tfidf, y_test, cv=strat_k_fold, scoring='precision_weighted', n_jobs=-1))
#accuracy= np.mean(cross_val_score(MLPClassifier(), X_test_tfidf, y_test, cv=strat_k_fold, scoring='accuracy', n_jobs=-1))
#print(score)
#print(accuracy)
#print(precision)
#print("")



