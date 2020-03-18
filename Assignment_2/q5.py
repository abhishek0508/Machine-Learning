import pandas as pd
import numpy as np
import io
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from yellowbrick.text import TSNEVisualizer
from sklearn.metrics import confusion_matrix

url='/home/abhishek/dev/Semester_2/SMAI/Assignments/Assignment_2/Datasets/Question-5/Train(1).csv'
test_url = '/home/abhishek/dev/Semester_2/SMAI/Assignments/Assignment_2/Datasets/Question-5/test.csv'

class AuthorClassifier:
    
    clf = None   
    vectorizer_object = None
    vectorizer = TfidfVectorizer(stop_words="english")

    def stemming(self,X):    
        X = X.to_numpy()
        doc = []
        for i in range(0,len(X)):
            X[i] = X[i].translate(str.maketrans('', '', string.punctuation + string.digits))
            stemmer = PorterStemmer()
            input_str = word_tokenize(X[i])
            result = ""
            for word in input_str:
                result = result+" "+stemmer.stem(word)
            doc.append(result)
        return doc

    def predict(self,url):
        df = pd.read_csv(url)
        X = df['text']
        # y_true = df['author']
        X = np.asarray(self.stemming(X))
        X = self.vectorizer_object.transform(X)
        y_predict = self.clf.predict(X)
        return y_predict#,y_true

    def train(self,url): 
        df = pd.read_csv(url)
        X = df['text']
        y = df['author']
        X = np.asarray(self.stemming(X))
        self.vectorizer_object = self.vectorizer.fit(X)
        X = self.vectorizer_object.transform(X)
        self.clf = svm.SVC(kernel="linear",decision_function_shape='ovr',C=1.0)
        self.clf.fit(X, y)


# auth_classifier = AuthorClassifier()
# auth_classifier.train(url) # Path to the train.csv will be provided
# y_predict,y_test = auth_classifier.predict(test_url) # Path to the train.csv will be provided
# print(accuracy_score(y_test,y_predict))







        





    



    
    
