import numpy as np
import random
import pandas as pd
import io
from collections import Counter
from sklearn.metrics import accuracy_score

# train_url='https://raw.githubusercontent.com/abhishek0508/Machine-Learning/master/Assignment_1/OneDrive_1_21-01-2020/q1/train.csv'
# test_url='https://raw.githubusercontent.com/abhishek0508/Machine-Learning/master/Assignment_1/OneDrive_1_21-01-2020/q1/test.csv'
# test_label_url='https://raw.githubusercontent.com/abhishek0508/Machine-Learning/master/Assignment_1/OneDrive_1_21-01-2020/q1/test_labels.csv'


class KNNClassifier:
  
  train_data=[]
  train_label=[]
  
  def fun(self,dist,k):
    knn = []
    for i in range(0,k):
      knn.append(dist[i])
    # print(knn)
    knn_label = []
    for i in range(0,len(knn)):
      knn_label.append(knn[i][1])
    knn_value = Counter(knn_label)
    res = knn_value.most_common(1)[0][0]
    # print(res)
    return res

  def train(self,url):
    df = pd.read_csv(url)
    self.train_data = df[df.columns[1:]].to_numpy()
    self.train_label = df[df.columns[:1]].to_numpy()

  def predict(self,url):
    labels = []
    labels.append(6)
    dist = []
    test = pd.read_csv(url).to_numpy()
    for row_test in test:
      i=0
      for row_train in self.train_data:
        row_diff = np.power((row_test-row_train),2)
        row_euclid_diff = int(np.sqrt(np.sum(row_diff)))
        print(row_euclid_diff)
        x = int(self.train_label[i])
        # print(x)
        dist.append((row_euclid_diff,x))
        i=i+1
      # print(dist)
      dist2 = sorted(dist, key=lambda x: x[0])
      labels.append(self.fun(dist2,3))
      dist.clear()
    return labels



## Code for Working on google colab

# knn_classifier = KNNClassifier()
# knn_classifier.train(train_url)
# predictions = knn_classifier.predict(test_url)
# test_labels = list()
# test_label = pd.read_csv(test_label_url)
# test_label = test_label.to_numpy()
# test_label_new = []
# for i in range(0,test_label.shape[0]):
#   test_label_new.append(int(test_label[i,:]))
# print(test_label_new)
# print(predictions)
# print (accuracy_score(test_label_new, predictions))
      
