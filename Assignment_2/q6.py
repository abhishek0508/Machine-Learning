import numpy as np
import pandas as pd
import io
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import random
import re
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics.cluster import homogeneity_score
 

url = '/home/abhishek/dev/Semester_2/SMAI/Assignments/Assignment_2/Datasets/Question-6/dataset/'
test_url = '/home/abhishek/dev/Semester_2/SMAI/Assignments/Assignment_2/Datasets/Question-6/test/'

class Cluster:
    X_train = None
    y_train = None
    main_cluster = None
    cluster_labels = None
    current_point = None
    vectorizer_object = None
    y_true_labels = None

    def preprocessing(self,X_raw):
        vectorizer = TfidfVectorizer(stop_words="english")  
        self.vectorizer_object = vectorizer.fit(X_raw)
        X = self.vectorizer_object.transform(X_raw)
        # print(vectorizer.get_feature_names())
        return X

    def preprocessing_test(self,X_raw):
        vectorizer = TfidfVectorizer(stop_words="english")  
        X = self.vectorizer_object.transform(X_raw)
        return X

    def read_files(self,url):
        files = os.listdir(url)
        # print(len(files))
        doc = []
        label = []
        for fname in files:
            x = re.split("_", fname)
            y = re.split("\.",x[1])
            label.append(int(y[0]))
            with open(url+fname,"r") as f:
                data = f.read().replace('\n','')
                data = data.translate(str.maketrans('', '', string.punctuation+string.digits))
                stemmer = PorterStemmer()
                input_str = word_tokenize(data)
                result = ""
                for word in input_str:
                    result = result+" "+stemmer.stem(word)
                doc.append(result)
        return label, doc

    def read_test_files(self,url):
        files = os.listdir(url)
        doc = []
        for fname in files:
            with open(url+fname,"r") as f:
                data = f.read().replace('\n','')
                data = data.translate(str.maketrans('', '', string.punctuation+string.digits))
                stemmer = PorterStemmer()
                input_str = word_tokenize(data)
                result = ""
                for word in input_str:
                    result = result+" "+stemmer.stem(word)
                doc.append(result)
        return doc
    
    def calculate_cluster_median(self,cluster):
        print("calculating cluster centroid........")
        X_train_raw = pd.DataFrame(self.X_train.todense())
        new_centroid = [[],[],[],[],[]]
        for i in range(0,len(cluster)):
            new_centroid[i].append(X_train_raw.iloc[cluster[i]].mean(axis=0))
        return new_centroid

    def intial_cluster_centroid(self):
        cluster_centroid = [[],[],[],[],[]]
        return cluster_centroid

    def initial_cluster_centroid_random(self):
        X_train_raw = self.X_train.todense()
        total_points = self.X_train.shape[0]
        val_range = total_points/5
        fi = random.randint(0,val_range)
        si = random.randint(val_range,val_range*2)
        ti = random.randint(val_range*2,val_range*3)
        fo = random.randint(val_range*3,val_range*4)
        ft = random.randint(val_range*4,val_range*5)        
        cluster_centroid = [X_train_raw[fi],X_train_raw[si],X_train_raw[ti],X_train_raw[fo],X_train_raw[ft]]
        return cluster_centroid

    
    def fit(self):
        X_train_raw = self.X_train.todense()
        self.current_point = self.initial_cluster_centroid_random()
        
        print("Fitting the model............")
        for l in range(1,20):
            cluster = [[],[],[],[],[]]
            for j in range(0,len(X_train_raw)):
                minimum_point = int(1e10)
                index = 0
                for i in range(0,5):
                    euclid_dist = np.sum(np.square(self.current_point[i]-X_train_raw[j]))
                    if(euclid_dist < minimum_point):
                        minimum_point = euclid_dist
                        index = i
                cluster[index].append(j)
            self.current_point = self.calculate_cluster_median(cluster)
            print("Cluster sizes........")
            # for i in range (0,5):
            #     print(len(cluster[i]))
        
        self.main_cluster = cluster
        # print(self.main_cluster)

        print("Model Fitting Done........")

    def label_clusters(self):
        self.cluster_labels = {1:None,2:None,3:None,4:None,5:None}

        print("Labelling of clusters started.........")

        for i in range(0,len(self.main_cluster)):
            my_dict = {1:0,2:0,3:0,4:0,5:0}
            for j in range(len(self.main_cluster[i])):
                my_dict[self.y_train[self.main_cluster[i][j]]] = my_dict[self.y_train[self.main_cluster[i][j]]]+1 
            
            # print(my_dict)
            max_label = -1e10
            index = 0
            for j in range(1,6):
                if(my_dict[j] > max_label):
                    max_label = my_dict[j]
                    index = j

            
            self.cluster_labels[i+1] = index

        print("Labels of cluster........")
        # print(self.cluster_labels)



    def __init__(self):
        label,doc = self.read_files(url)
        # print(len(label))
        # print(len(doc))
        print("File Reading Done")
        self.X_train = doc
        self.y_train = label
        self.X_train = self.preprocessing(doc)
        print("Preprocessing done")
        # print(self.X_train.shape)
        self.fit()
        self.label_clusters()
        
    def train(self):
        return 0

    def cluster(self,url):
        print("predictions...................")
        predictions = []
        X_test = self.read_test_files(url)
        X_test = self.preprocessing_test(X_test)
        self.y_true_labels,docs_true = self.read_files(url)
        # print(X_test.shape)

        X_test_raw = X_test.todense()
        
        predictions = []
        for j in range(0,len(X_test_raw)):
            min_dist = 1e10
            index = 0
            for i in range(0,5):
                euclid_dist = np.sum(np.square(self.current_point[i]-X_test_raw[j]))
                if(euclid_dist < min_dist):
                    min_dist = euclid_dist
                    index = i+1
            predictions.append(index)
        return predictions

# cl = Cluster()
# cl.train()
# predictions = cl.cluster(test_url)
# print(predictions)
# print("Score == ")
# print(homogeneity_score(predictions,cl.y_true_labels))